# Transformerについて勉強したことまとめ

NLPに素人として、まだ至らない点も多いが、これまで自分の理解を整理します。

個人的に、Transformerの運用に興味を持つので、下記の疑問点について知りたくて勉強を始まりました。

- どんなデータを食う？
- Attentionの構造は？
- どんな結果を吐く？

これら点について、自分の理解を纏めます。  

## Transformerモデル

Transformerについてすべての知識の紹介を私の能力範囲外です。詳しく情報が欲しかったら、下記の文章の参考してください。  
[自然言語処理の必須知識 Transformer を徹底解説！](https://deepsquare.jp/2020/07/transformer/)  

説明便宜のために、まずTransformerモデルの全体図を張ります。　　

![tranformer model image](https://deepsquare.jp/wp-content/uploads/2020/07/pasted-image-0-1.png)  

説明の例として、下記のタスクとします。  
「あれはロボットです」  
を  
「That is a robot」  
に翻訳する

## __インプット__

Transformerモデルにインプットするデータについて、下記の概念が知る必要です。

- Sequence
- Input to Encoder
- Input to Decoder
- Tokenization
- Positional Embedding

### __*Sequence*__

TransformerはSeq2Seq構造の一種であり、インプットするデータはよくSequenceといいます。これは「順序的なデータ」とのことですね。
例えば、言葉は「順序的な単語」であり、画像は「順序的なピクセル」であります。単語の順／ピクセルの順が変わると意味・画面も変わりますよね。  

NLPの場合、一つSequenceは一つセンテンスであります。「分類AI」などと違うのは、Transformerにインプットするデータは、一つではなく、EncoderとDecoderにそれぞれのSequenceをインプットする必要であります。  
※　Transformer構造図に、Decoderへのデータは「Outputs」と書かれています。ちょっと微妙な表現と思います。個人的に、計算後のデータをアウトプットであり、計算用データはインプットと思っていますので、この「Outputs」もインプットと呼ばせてください。

### __*Encoder Inputs*__

Encoderにインプットするのは「あれはロボットです」。

### __*Decoder Inputs*__

Decoderのインプットは二つソースがあります：

- 「That is a robot」　※
- Encoderの算出結果

※　訓練時のインプットである。推論時に、ここには開始記号（&lt;BOS&gt;）しかない。（Tokenization節に説明します）  
※　訓練結果チェック用データであるため、Transformer構造図にはこれを「Outputs」と呼びます（？）

Transformer構造図にEncoderのアウトプットはDecoderのインプットとして渡されています。

また一つのインプットがあり、翻訳結果の「That is a robot」です。

### __*Tokenization*__

Transformerに上記のように直接に言葉を入れても言葉の処理ができない。言葉はベクターに転換する必要です。
<pre>
１．Encoder側もDecoder側もそれぞれの「辞書」を持っている
２．インプットのセンテンスの単語は、それぞれの辞書に対応するインデックスがある
３．「あれ」「は」「ロボット」「です」は辞書インデックスに変更
４．変更後、[9, 5, 45, 10]のような配列になる
５．すべてのセンテンスの単語数が同じではないので、最大の数を想定、Max_Sequence_Lengthを決める。４の配列はMax_Sequence_Lengthの配列に整形する。例えば、最大６単語の場合、[9, 5, 45, 10, &lt;EOS&gt;, &lt;EOS&gt;]にする
</pre>
&lt;EOS&gt;は「終了記号」の辞書インデックスであります。

もし、あるセンテンスの長さが「Max_Sequence_Length - 1」を超えた場合、このセンテンスを「Max_Sequence_Length - 1」まで切り取り、最後に&lt;EOS&gt;を付けます。

トーケン化の段階、また一つやるべきことがある --> Decoderのインプットデータの先頭に開始記号&lt;BOS&gt;の追加する必要があります。

Transformer構造図のDecoderの下に「Outputs(shifted right)」と書かれています。「Shifted right」とは、インプットSequenceの左に"開始"記号を追加することです。  

よって、DecoderにインプットするSequenceは実際に「&lt;BOS&gt; That is a robot」のようになります。  

なぜ開始記号（&lt;BOS&gt;）が必要ですか？Decoderの実行は下記のようになっています：
<pre>
０．Encoderの結果が算出された
１．Encoderの結果と、DecoderはインプットSequenceの一番目の単語（&lt;BOS&gt;）を使って一番目の翻訳結果単語を算出 (That)
２．次は、Encoderの結果と（&lt;BOS&gt; That）で、次の単語を算出 (is)
３．次は、Encoderの結果と（&lt;BOS&gt; That is）で、次の単語を算出 (a)
４．次は、Encoderの結果と（&lt;BOS&gt; That is a）で、次の単語を算出 (robot)
５．&lt;EOS&gt;まで上記のように繰り返し、翻訳作業となります。
</pre>
訓練の時にはDecoderのインプットSequenceは「標準結果」として使い、Layersの重さの調整を行います。推論の時には開始記号とEncoderの結果を使って、推論の開始ができます。

Tokenizationで言葉をベクターに変換後、EmbeddingとPosition Encodingをします。

### __*Positional Embedding*__

Attentionにインプットするのは、トーケンしたSequenceではなく、Word Embedding と Positional Encoding 処理が必要です。

- Embedding
  （[Why do we use word embeddings in NLP?](https://towardsdatascience.com/why-do-we-use-embeddings-in-nlp-2f20e1b632d2)）  
  
  Embedding処理は、Tensorflowのkeras.layers.Embeddingを使って実現できます。EmbeddingのDimentionは任意であるが、NLPの場合、EmbeddingのDimentionは「512」にします。

  よって、先ほどの[9, 5, 45, 10]を例とします：  
  Embedding 前  
  <pre>
  [あれ]      [9]
  [は]        [5]
  [ロボット]  [45]
  [です]      [10]
  </pre>

  Embedding 後  
  <pre>
  [あれ]      [0.12 0.001 .... 0.12]    (512個)
  [は]        [0.02 0.01 .... 0.021]　  (512個)
  [ロボット]  [0.1 0.23 .... 0.003]     (512個)
  [です]      [0.031 0.0044 .... 0.02]  (512個)
  </pre>

- Positional Ecoding
  
  Attentionにインプットする前、またPositional Encodingも必要であります。
  
  一つ言葉に、単語の位置が違うと、意味が全く違うことが多いですね。位置情報も導入します。  

  下記の公式で位置情報を算出します。  
  ![positional encoding](./res/positional_encoding.png)  
  中に、  
  ・ d_modelは、EmbeddingのDimentionであります（先の例に512）。  
  ・ 2i は偶数行、2i+1 は奇数行を表します。

  Positional Encodingは位置情報の計算のみなので、直接にインプットのベクターを使わない。  

  そして、Attentionにインプットするデータができます：  
  Positional Embedding = Embedding結果 + Posintal Encoding結果

## __Self-Attention__

TransformerのAttention構造は下記図のようになっています。同じインプットデータでK,V,Q 三つLayerにも渡すので、「Self」Attentionといいます。

![Self-Attention図](https://deepsquare.jp/wp-content/uploads/2020/07/pasted-image-0-2.png)

インプットデータは「Positional Embedding」処理済みデータであります。バッチを考えしなければ、下記のようなデータとなっています。  
・　[max_sequence_szie, embeded_dimension]  
リアル世界だと、  
・　[batch_size, max_sequence_size, embeded_dimension]となります。  
バッチデータの場合でも、毎回計算に入れたのは、後ろの２次元のデータです。  

Self-Attentionの計算過程が複雑ではない、図に示したように、Q、K は 「ドット積」の結果をLogit -> Softmax など計算を行い、結果をV のデータともう一回「ドット積」算を実施します。算出した結果はこのAttentionのアウトプットとして、次のAttentionのインプットデータとして渡します。  

具体的な計算は、learning_transformer_transformer_tensorflow.ipynb の「SelfAttention」クラスの実装を参照してください。

EncoderとDecoderはそれぞれ複数Attentionを構成しています。注意すべきのは、Decoderの第二層のAttentionのインプットデータの K、V はEncoderのアウトプットであること。

Attentionの計算結果も[batch_size, max_sequence_size, embeded_dimension]の形になります。

## __MultiHead Attention__

MultiHead Attentionの複数Attentionが並んで構成したことです。実際の作りは下記のようになっています。

<pre>
０．例えば元々AttentionへのインプットTensorは [16, 30, 512] の形であります。
   （10: batch_size, 32: max_sequence_length, 512: embed_dimension）
１．d_model = Embed_Dimmension / Head_Num で Attention_Dimension (d_model) を決める。
    例えばHead_Num = 8であれば、d_model = 512/8 = 64 になります。
２．そしてMultiHead Attentionは[16, 8, 32, 64]の形にReshapeにします。
   （batch_size, head_num, seq_length, d_model）  
３．Attentionは、最後二次元（32, 64）で使って計算しますので、MultiHeadにReshape後にも元々の計算に影響がないです。
４．Attentionの計算が終わったら、transposeで、Attentionのアウトプットを[batch_size, seq_length, head_num, d_model]に整形します。
５．再度Reshapeで[batch_size, seq_length, embed_dimension]に整形すると、元の形に戻します。
</pre>

MultiHeadにするのは、二つメリットがあります。  
１．同じデータでも複数（Head数）の特徴の抽出することができます。  
２．複数Headが平行実行ができます。

## __Encoder & Decoder__

まだ実装してみてないので、上にもう紹介した関連部分以外、実装レベルの詳しく説明できません。

## __アウトプット__

インプットの説明に、EncoderとDecoderのアウトプットについてちょっと説明しましたが、まだ実装してみてないので、詳しく説明できません。
