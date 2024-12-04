import torch
from argparse import Namespace  # Namespace를 import
from torchvision import transforms
from PIL import Image
from encoder4editing.models.psp import pSp  # e4e 모델 가져오기
import os

# e4e 모델 체크포인트 경로
e4e_checkpoint_path = "/home/yjnoh/workspace/CMU_IDL_Project/e4e_ffhq_encode.pt"

# 저장 경로 설정
output_dir = "./latent_vectors/"
os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 장치 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# e4e 모델 로드 함수
def load_e4e_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts['return_latents'] = True  # 명시적으로 설정
    opts = Namespace(**opts)  # Namespace 객체로 변환
    e4e_model = pSp(opts)
    e4e_model.eval().to(device)

    print("e4e Model Options:", opts)  # 모델 옵션 출력
    return e4e_model

# e4e 모델 로드
e4e_model = load_e4e_model(e4e_checkpoint_path)

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # e4e는 256x256 이미지를 처리
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # -1 ~ 1로 정규화
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Latent Vector 생성 함수
# def encode_with_e4e(e4e_model, image_tensor):
#     with torch.no_grad():
#         outputs = e4e_model(image_tensor, randomize_noise=False, return_latents=True)
#         latent_code = outputs[0]  # \( W+ \) 공간의 잠재 벡터 반환
#         print("Latent vector shape:", latent_code.shape)
#     return latent_code

def encode_with_e4e(e4e_model, image_tensor):
    with torch.no_grad():
        _, latent_code = e4e_model(image_tensor, randomize_noise=False, return_latents=True)
        print("Latent vector shape:", latent_code.shape)  # 디버깅용 출력
    return latent_code

# Latent Vector 저장 함수
def save_latent_vector(latent_vector, output_path):
    torch.save(latent_vector, output_path)
    print(f"Latent vector saved to {output_path}")

# 이미지 경로
image_path = "/home/yjnoh/workspace/CMU_IDL_Project/self-portrait-1955.jpg"

# 1. 이미지 전처리
image_tensor = preprocess_image(image_path)

# 2. e4e로 Latent Vector 생성
latent_vector = encode_with_e4e(e4e_model, image_tensor)

# \( W+ \) 공간으로 생성된 잠재 벡터 확인
print("Latent vector shape:", latent_vector.shape)  # 예상: [1, 18, 512]

# 3. Latent Vector 저장
output_path = os.path.join(output_dir, "latent_vector.pt")
save_latent_vector(latent_vector, output_path)

