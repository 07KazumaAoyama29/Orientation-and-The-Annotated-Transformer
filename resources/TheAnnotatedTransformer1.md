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
提案された当初は、英語→ドイツ語の翻訳といった自然言語処理の系列変換タスクでのみ注目されていたが、今日ではChatGPTをはじめとした様々なタスクやモダリティに応用され、その有用性が広く認識されている。<br>
### "The Annotated Transformer" [1][tran] とは?
上記で述べたとおり、 Transformer の最も重要な構成要素は"Attention"です。<br>
その"Attention"の仕組みを論文"Attention is All You Need"[2]に基づいて、コードを交えて解説している良い資料です。<br>
### 背景
Transformerの最大の特徴は、系列変換の並列計算の高速化です。少し難しい言い方をすると、畳み込みニューラルネットワーク（CNN）を基本構成要素として使用し、入力と出力のすべての位置に対して隠れた表現を並列に計算します。これにより、 Extended Neural GPU, ByteNet, ConvS2S の基盤を成しています。<br>
Attentionの構成要素には、Self Attention, Multi Head Attention みたいな難しいのもありますが、ここでは割愛♡します。<br>

### Part1: モデル構造 Model Architecture
#### 簡単な説明
最も強力な(2023年では)自然言語処理や音声処理などで使われるモデル構造に、"エンコーダ・デコーダ構造"[3]があります。<br><br>
ここで、"エンコーダ"とは、入力シンボル表現のシーケンス （x1​，...，xn​） を連続表現のシーケンス z = （z1​，...，zn​） にマッピングするものです。通常、入力されるシンボルは離散的なもので、単語や文字などが例として挙げられます。<br>マッピングされた連続表現のシーケンス z は、ニューラルネットワークの隠れ層での計算に使われます。通常のシンボルの状態だと、ニューラルネットの計算に使いにくいので、計算しやすいベクトル表現にマッピングするみたいなイメージです。<br><br>
"デコーダ"ではzが与えられると、シンボルの出力シーケンス（y1​，...，ym​）を1要素ずつ生成します。デコーダの出力が、最終的な出力になります。<br><br>
要するに、エンコーダは入力を「解釈」し、デコーダはその解釈を基に出力を「生成」する役割を果たします。<br><br>
Transformerの全体図は、https://nlp.seas.harvard.edu/annotated-transformer/ のPart1を参照してください。
#### program: モデル構造のクラス
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
#### 多層エンコーダ
Self Attention という言葉を聞いたことはありますか？<br>
Self Attention はシーケンス内の各要素の依存関係を捉える事を目的としています<br>
単層ではシーケンスの処理を左から右に線形に行うため、シーケンス内の各要素の依存関係を捉えることができないが、多層にすることで並列計算が可能となり、それぞれの計算結果を別のタスクに割り当てることによって、各要素の依存関係を捉える事を実現している。<br>

#### 多層エンコーダの構成要素
多層エンコーダの各層には、２つのサブレイヤーがあり、"Self Attention"と"Feedforward Network"と呼ばれている。<br>
以下では、大きい実装から小さい実装を導いていく。まずはエンコーダのクラスを作り、各層のエンコーダを作り、２つのサブレイヤー"Self Attention"と"Feedforward Network"をつくる。次にそれらを合成するクラスを作り、多層エンコーダの実装を行う。

#### program: 多層エンコーダ
ここでは、多層エンコーダの層の数を6とする。理由は知りません。<br>
```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):#おそらく、左から順番にシーケンスを処理する部分
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
#### design of multilayer encoder
単層エンコーダで処理する場合は、計算結果は左から右に伝播されていく。<br>
エンコーダを多層化するためには、左から伝播されていく情報に加えて、別のレイヤーの情報も受け取ることになる。<br><br>
ただ単に足し合わせるのではなく、残差接続(Residual Connection)[4] と呼ばれる手法を用いてそれらを合成処理することによって、深いネットワークでの勾配消失を防ぎ、情報が層を越えて伝播しやすくなる。<br>
また、残差接続をした後にレイヤー正規化(Layer Normalization)[5]という操作が行われる。<br>
レイヤー正規化は、各層の出力を正規化し、学習を安定させる役割を果たす。これにより、勾配の変動を抑え、学習がスムーズに行われるようになる。<br><br>
上記の操作を、LayerNormと呼ぶ(多分)<br>

#### program: LayerNorm
```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
#### 残差接続の計算
上記では、同一層から伝播されてきた情報と、他のレイヤーの情報をただ単に足し合わせるのではなく、残差接続(Residual Connection)[4] と呼ばれる手法を用いてそれらを合成処理することによって、深いネットワークでの勾配消失を防ぎ、情報が層を越えて伝播しやすくなる。と書きました。<br>
さらに、ドロップアウト[6]という手法があります。これは過学習を防ぐために使用され、学習時にニューロンをランダムに無効化することによって、モデルの汎化能力を高めます。

#### program: add dropout
```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```
#### program: complete multilayer encoder
```python
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```
## 参考文献
[1] Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman(2022). The Annotated Transformer. https://nlp.seas.harvard.edu/annotated-transformer/<br>
[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser(2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30 (NIPS 2017)<br>
[3] Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio. NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE(2014). https://arxiv.org/abs/1409.0473<br>
[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385<br>
[5] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. Layer Normalization. https://arxiv.org/abs/1607.06450
[6] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. https://jmlr.org/papers/v15/srivastava14a.html


[3] 森下篤(2024). Visual Studio Code 実践ガイド. 技術評論社<br>
[4] Bill Ludanovic, 鈴木駿, 長尾高弘(2022). 入門 Python3 第二版. O'Reilly Japan<br>
[5] Al Sweigart, 相川愛三(2023). 退屈なことはPythonにやらせよう　第二版. O'Reilly Japan<br>
[6] Al Sweigart, 岡田祐一(2022). きれいなPythonプログラミング. マイナビ<br>

This material benefited from the assistance of ChatGPT.

Kazuma Aoyama(kazuma-a@lsnl.jp), Yoshiteru Taira(yoshiteru@lsnl.jp)
[tran]: https://nlp.seas.harvard.edu/annotated-transformer/