## 2.The Annotated Transformer
### pythonの環境構築(仮想環境推奨)
#### "venv"という名前の仮想環境を作成 #pythonのバージョンは3.9にしてください。
```bash
py -3.9 -m venv venv 
```
#### 仮想環境のアクティベート
- Linux, Mac
```bash
source ./venv/bin/activate
```
- Windows
```bash
source ./venv/Scripts/activate
```
#### 必要なmoduleのインストール #※めっちゃ時間かかります
```bash
pip install -r requirements.txt 
```
#### 仮想環境のディアクティベート
```bash
deactivate
```
### 内容目標
### Transformer
Transformer とは、系列情報を処理する、注意機構(Attention)を主体とした深層学習モデルです。<br>
提案された当初は、英語→ドイツ語の翻訳といった自然言語処理の系列変換タスクでのみ注目されていたが、今日ではChatGPTをはじめとした様々なタスクやモダリティに応用され、その有用性が広く認識されています。<br>
### "The Annotated Transformer"[1]とは?
上記で述べたとおり、 Transformer の最も重要な構成要素は"Attention"です。<br>
その"Attention"の仕組みを論文"Attention is All You Need"[2]に基づいて、コードを交えて解説している良い資料です。<br>
### 背景
Transformerの最大の特徴は、系列変換の並列計算の高速化です。少し難しい言い方をすると、畳み込みニューラルネットワーク（CNN）を基本構成要素として使用し、入力と出力のすべての位置に対して隠れた表現を並列に計算します。これにより、 Extended Neural GPU, ByteNet, ConvS2S の基盤を成しています。<br>
Attentionの構成要素には、Self Attention, Multi Head Attention みたいな難しいのもありますが、ここでは割愛♡します<br>

### Part1: モデル構造 Model Architecture
#### 簡単な説明
最も強力な(2023年では)自然言語処理や音声処理などで使われるモデル構造に、"エンコーダ・デコーダ構造"[3]があります。<br>
ここで、"エンコーダ"とは、入力シンボル表現のシーケンス （x1​，...，xn​） を連続表現のシーケンス z = （z1​，...，zn​） にマッピングするものです。通常、入力されるシンボルは離散的なもので、単語や文字などが例として挙げられます。マッピングされた連続表現のシーケンス z は、ニューラルネットワークの隠れ層での計算に使われます。通常のシンボルの状態だと、ニューラルネットの計算に使いにくいので、計算しやすいベクトル表現にマッピングするみたいなイメージです。<br>
"デコーダ"ではzが与えられると、シンボルの出力シーケンス（y1​，...，ym​）を1要素ずつ生成します。デコーダの出力が、最終的な出力になります。<br>
要するに、エンコーダは入力を「解釈」し、デコーダはその解釈を基に出力を「生成」する役割を果たします。<br>
#### programn: モデル構造のクラス
```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
```




## 参考文献
[1] Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman(2022). The Annotated Transformer. https://nlp.seas.harvard.edu/annotated-transformer/<br>
[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser(2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30 (NIPS 2017)<br>
[3]Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio. NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE(2014). https://arxiv.org/abs/1409.0473<br>

[3] 森下篤(2024). Visual Studio Code 実践ガイド. 技術評論社<br>
[4] Bill Ludanovic, 鈴木駿, 長尾高弘(2022). 入門 Python3 第二版. O'Reilly Japan<br>
[5] Al Sweigart, 相川愛三(2023). 退屈なことはPythonにやらせよう　第二版. O'Reilly Japan<br>
[6] Al Sweigart, 岡田祐一(2022). きれいなPythonプログラミング. マイナビ<br>

This material benefited from the assistance of ChatGPT.

Kazuma Aoyama(kazuma-a@lsnl.jp), Yoshiteru Taira(yoshiteru@lsnl.jp)