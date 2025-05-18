import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import BertTokenizer

# Define the ResNet feature extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define the cross-attention mechanism 
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, y):
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        return out

# Define main VQAModel
class VQAModel(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=768):
        super().__init__()
        self.resnet_features = ResNetFeatureExtractor(output_dim=hidden_dim)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_projection = nn.Linear(768, hidden_dim)
        self.fusion = CrossAttention(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, question_ids, attention_mask):
        image_features = self.resnet_features(image)
        question_output = self.bert(question_ids, attention_mask=attention_mask)
        question_features = self.bert_projection(question_output.last_hidden_state)

        fused_features = self.fusion(question_features, image_features.unsqueeze(1))
        attention_weights = self.attention(fused_features)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_features = (fused_features * attention_weights).sum(dim=1)

        logits = self.classifier(attended_features)
        return logits

# Function to preprocess input image
def preprocess_image():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to preprocess input question
def preprocess_question(question: str, max_length: int = 64):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(
        question,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    return tokens
