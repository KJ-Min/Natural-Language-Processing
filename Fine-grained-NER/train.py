import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from seqeval.metrics import f1_score
import random


from utils import get_dataloader

def set_seed(seed_value):
    """재현성을 위한 시드 설정"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def train_epoch(model, dataloader, optimizer, scheduler, device, clip_norm):
    """한 epoch 동안 모델을 학습하는 함수"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        """train_dataloader에서 배치를 가져와 반복"""
        input_ids = batch['input_ids'].to(device) # 입력 ID를 GPU로 이동
        attention_mask = batch['attention_mask'].to(device) # 어텐션 마스크를 GPU로 이동
        labels = batch['labels'].to(device) # 레이블을 GPU로 이동

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # 모델에 Input을 전달하여 Output(Loss 포함) 계산
        loss = outputs.loss # 모델 Output에서 Loss 가져오기
        
        loss.backward() # 그래디언트 계산

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm) # 그래디언트 클리핑 적용하여 exploding gradients 방지
        optimizer.step() # 모델 파라미터 업데이트
        scheduler.step() # 스케줄러 업데이트
        optimizer.zero_grad() # 그래디언트 초기화

        total_loss += loss.item() # Loss 누적

    return total_loss / len(dataloader) # Epoch의 평균 학습 Loss 반환


def evaluate_epoch(model, dataloader, device, id2label_map):
    """한 epoch 동안 모델을 평가하는 함수"""
    model.eval() 
    total_loss = 0 
    all_preds = [] # 모든 예측값 저장할 리스트
    all_labels = [] # 모든 실제 레이블 저장할 리스트

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for batch in tqdm(dataloader, desc="Evaluating"):
            """validation_dataloader에서 배치를 가져와 반복"""
            input_ids = batch['input_ids'].to(device) # 입력 ID를 GPU로 이동
            attention_mask = batch['attention_mask'].to(device) # 어텐션 마스크를 GPU로 이동
            labels = batch['labels'].to(device) # 레이블을 GPU로 이동

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # 모델에 Input을 전달하여 Output(Loss 포함) 계산
            loss = outputs.loss # 모델 Output에서 Loss 가져오기
            logits = outputs.logits # 모델 Output에서 logits(분류 점수) 가져오기

            total_loss += loss.item() # Loss 누적
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy() # logits에서 가장 높은 값을 가진 인덱스를 예측 레이블로 변환 (CPU로 이동 후 NumPy 배열로 변환)
            true_labels = labels.cpu().numpy() # 실제 레이블을 CPU로 이동 후 NumPy 배열로 변환

            for i in range(len(true_labels)):
                """배치 내 각 샘플에 대해 반복"""
                pred_seq = [] # 현재 샘플의 예측 Label 시퀀스
                label_seq = [] # 현재 샘플의 실제 Label 시퀀스
                for pred_id, label_id in zip(predictions[i], true_labels[i]):
                    """예측 ID와 실제 ID를 하나씩 비교"""
                    if label_id != -100: # 실제 Label ID가 -100(None)이 아니면
                        pred_seq.append(id2label_map.get(pred_id, "O")) # 예측 ID를 Label 문자열로 변환 (ID가 없으면 "O"로 처리)
                        label_seq.append(id2label_map.get(label_id, "O")) # 실제 ID를 Label 문자열로 변환 (ID가 없으면 "O"로 처리)
                if pred_seq and label_seq: # 예측 시퀀스와 실제 시퀀스가 모두 비어있지 않으면
                    all_preds.append(pred_seq) # 전체 예측 리스트에 추가
                    all_labels.append(label_seq) # 전체 실제 레이블 리스트에 추가
    
    avg_loss = total_loss / len(dataloader) # Epoch의 평균 검증 Loss 계산
    
    if not all_preds or not all_labels: # 예측값이나 실제 레이블이 없는 경우 F1 계산을 위한 경고 출력
        print("Warning: No valid predictions or labels found for F1 calculation.")
        micro_f1 = 0.0 # 마이크로 F1 스코어를 0.0으로 설정
    else: # 예측값과 실제 레이블이 모두 있는 경우
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0) # 마이크로 F1 스코어 계산 (0으로 나누는 경우 0으로 처리)

    return avg_loss, micro_f1 # 평균 검증 Loss, 마이크로 F1 스코어 반환


def main(args):
    """메인 학습 로직 함수"""
    set_seed(args.seed) # 시드 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """레이블 맵 파일 로드"""
    try:
        with open(os.path.join(args.label_map_path, f"fine_id2label.json"), "r") as f: # id2label.json 파일 열기
            id2label = json.load(f) # JSON 파일 로드하여 id2label 딕셔너리 생성
        id2label = {int(k): v for k, v in id2label.items()} # 딕셔너리의 키를 정수형으로 변환

        with open(os.path.join(args.label_map_path, f"fine_label2id.json"), "r") as f: # label2id.json 파일 열기
            label2id = json.load(f) # JSON 파일 로드하여 label2id 딕셔너리 생성
    except FileNotFoundError: # 파일이 없을 경우 예외 처리
        print(f"Error: Label map files not found at {args.label_map_path}. Exiting.")
        return # 함수 종료
    
    num_labels = len(id2label) # Label 개수 계산
    print(f"Number of labels: {num_labels}")
    if num_labels != 133: # Label 개수가 133이 아니면 경고 출력
        print(f"Warning: Expected 133 labels, but found {num_labels}. Check your label map files.")

    """토크나이저 로드"""
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    """모델 로드"""
    print(f"Loading model: {args.model_name_or_path} for token classification with {num_labels} labels.")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path, # 모델 이름 또는 경로
        num_labels=num_labels # Label 개수 설정
    )
    model.to(device) # 모델을 GPU로 이동

    """레이어 동결"""
    if args.freeze_layers: # 레이어 동결 옵션이 True이면 수행
        print(f"Freezing first {args.num_frozen_layers} encoder layers of ELECTRA.")
        
        """Embedding 레이어 동결"""
        for param in model.electra.embeddings.parameters(): # 임베딩 레이어의 모든 파라미터에 대해 반복
            param.requires_grad = False # 그래디언트 계산 비활성화 (동결)
        
        """지정된 수만큼의 초기 Encoder 레이어 동결"""
        if args.num_frozen_layers > 0 and args.num_frozen_layers < len(model.electra.encoder.layer): # 동결할 레이어 수가 유효한 범위(ELECTRA-base:12) 내에 있으면 수행
            for layer_idx in range(args.num_frozen_layers): # 지정된 수만큼 반복
                for param in model.electra.encoder.layer[layer_idx].parameters(): # 해당 인코더 레이어의 파라미터에 대해 반복
                    param.requires_grad = False # 그래디언트 계산 비활성화 (동결)
        elif args.num_frozen_layers >= len(model.electra.encoder.layer): # 동결할 레이어 수가 전체 인코더 레이어 수보다 크거나 같으면 (잘못된 설정)
            print(f"Warning: num_frozen_layers ({args.num_frozen_layers}) is too high. Freezing all encoder layers except classifier.")
            for layer_idx in range(len(model.electra.encoder.layer)): # 모든 인코더 레이어에 대해 반복
                for param in model.electra.encoder.layer[layer_idx].parameters(): # 해당 인코더 레이어의 파라미터에 대해 반복
                    param.requires_grad = False # 그래디언트 계산 비활성화 (동결)

    """Train DataLoader 생성"""
    print("Creating train dataloader...")
    train_dataloader = get_dataloader(
        split="train", # 데이터셋 파일 이름 (train.json)
        tokenizer=tokenizer, # 토크나이저
        label2id=label2id, # label to ID 딕셔너리
        id2label=id2label, # ID to label 딕셔너리
        batch_size=args.batch_size, # 배치 크기
        max_length=args.max_length, # 최대 시퀀스 길이
        json_base_path=args.data_dir, # JSON 데이터 파일 기본 경로
        shuffle=True # 데이터 셔플 여부 (학습 시 True)
    )

    """Validation DataLoader 생성"""
    print("Creating validation dataloader...")
    valid_dataloader = get_dataloader(
        split="validation", # 데이터셋 파일 이름 (validation.json)
        tokenizer=tokenizer, # 토크나이저
        label2id=label2id, # label to ID 딕셔너리
        id2label=id2label, # ID to label 딕셔너리
        batch_size=args.batch_size, # 배치 크기
        max_length=args.max_length, # 최대 시퀀스 길이
        json_base_path=args.data_dir, # JSON 데이터 파일 기본 경로
        shuffle=False # 데이터 셔플 여부 (검증 시 False)
    )

    """AdamW Optimizer 생성"""
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # 학습 가능한 파라미터만 전달 (requires_grad=True 인 파라미터)
        lr=args.learning_rate, # 학습률
        weight_decay=args.weight_decay, # Weight Decay 적용하여 Overfitting 방지
        eps=1e-8 # AdamW의 epsilon 값
    )

    """Step 수 계산"""
    total_training_steps = len(train_dataloader) * args.epochs # 총 학습 스텝 수 계산
    num_warmup_steps = int(total_training_steps * args.warmup_ratio) # 웜업 스텝 수 계산

    """Linear Scheduler 생성"""
    scheduler = get_linear_schedule_with_warmup(
        optimizer, # 옵티마이저
        num_warmup_steps=num_warmup_steps, # 웜업 스텝 수
        num_training_steps=total_training_steps # 총 학습 스텝 수
    )

    """Training 시작"""
    best_val_f1 = 0.0 # Best Validation F1 스코어
    patience_counter = 0 # Early Stopping을 위한 Patience 카운터

    print("Starting training...")
    for epoch in range(args.epochs):
        """지정된 Epochs만큼 반복"""
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        """1 Epoch Training"""
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, args.gradient_clipping_norm
        )
        print(f"Average Training Loss: {train_loss:.4f}") # 평균 학습 Loss 출력

        """1 Epoch Validation"""
        val_loss, val_f1 = evaluate_epoch(
            model, valid_dataloader, device, id2label
        )
        print(f"Average Validation Loss: {val_loss:.4f}") # 평균 검증 Loss 출력
        print(f"Validation F1 (micro): {val_f1:.4f}") # 검증 F1 스코어 출력

        """Early Stopping"""
        if val_f1 > best_val_f1: # 현재 검증 F1 스코어가 최고 F1 스코어보다 높으면 수행
            best_val_f1 = val_f1 # 최고 F1 스코어 업데이트
            patience_counter = 0 # patience 카운터 초기화
            print(f"New best F1: {best_val_f1:.4f}. Saving model...")
            os.makedirs(args.output_dir, exist_ok=True) # 출력 디렉토리 생성 (이미 존재하면 무시)
            model.save_pretrained(args.output_dir) # 모델 저장
            tokenizer.save_pretrained(args.output_dir) # 토크나이저 저장
            with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f: # Training arguments 저장을 위해 파일 열기
                json.dump(vars(args), f, indent=4) # Training arguments를 JSON 형태로 저장

        else: # 현재 검증 F1 스코어가 최고 F1 스코어보다 높지 않으면 수행
            patience_counter += 1 # patience 카운터 증가
            print(f"Validation F1 did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience: # patience 카운터가 설정된 임계값 이상이면 수행
                print("Early stopping triggered.")
                break # 학습 루프 종료
        
    print("\nTraining finished.")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-grained NER Model Training Script")
    
    """Paths and Model"""
    parser.add_argument("--model_name_or_path", type=str, default="google/electra-base-discriminator") # 모델 이름/경로
    parser.add_argument("--data_dir", type=str, default="data/files/") # 데이터 디렉토리
    parser.add_argument("--label_map_path", type=str, default="data/bio_label_ids") # 레이블 경로
    parser.add_argument("--output_dir", type=str, default="checkpoints") # 출력 디렉토리

    """Training Hyperparameters"""
    parser.add_argument("--epochs", type=int, default=500) # Epochs
    parser.add_argument("--batch_size", type=int, default=32) # 배치 크기
    parser.add_argument("--max_length", type=int, default=256) # 최대 시퀀스 길이
    parser.add_argument("--learning_rate", type=float, default=1e-5) # 학습률
    parser.add_argument("--weight_decay", type=float, default=0.01) # Weight Decay
    parser.add_argument("--warmup_ratio", type=float, default=0.1) # 웜업 비율
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0) # 그래디언트 클리핑 norm 값
    parser.add_argument("--early_stopping_patience", type=int, default=2) # 조기 종료 patience
    
    """Layer Freezing options"""
    parser.add_argument("--freeze_layers", type=bool, default=False) # 레이어 동결 여부
    parser.add_argument("--num_frozen_layers", type=int, default=9) # 동결할 레이어 수 (ELECTRA-base Last 3 Layer만 학습)

    """Seed"""
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    print("Training Arguments:")
    print(json.dumps(vars(args), indent=2))

    main(args) # 메인 함수 호출 (파싱된 인자 전달)
