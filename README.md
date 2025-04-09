# 2024-04-15勉強会<br>オリエンテーション + The Annotated Transformer①
## 概要
- 2024-04-15実施の勉強会についての資料です。<br>
- 基本的にこの資料の上から順番に進めていきます。<br>
- programフォルダには勉強会でコーディング予定のプログラムを置いているので、適宜参照してください。<br>

## 0.勉強会に関係ない諸連絡
研究テーマ考案ミーティング、研究ミーティング(Unixを楽しむ会と勉強会も?)はlsnlのZoomで行います。<br>
ZoomのIDやパスコード等の諸々の情報は下記のredmineの研究室運営のwikiに書いてあるので、そこを参照してください。<br>
https://rm.lsnl.jp/projects/adm/wiki

## 1.勉強会についてのオリエンテーション
### 勉強会とは
- 特定のテーマについて、参加者が学びながら知識を深めることを目的として行います。<br>
- テーマはその日の発表者に決めてもらいます。<br>
研究に関係あることでも無い事でも大丈夫です。情報系に関係のないことでも勿論大丈夫です。<br>
去年の勉強会では、ドメインの取得やWebページ公開、GitHubハンズオン、就活の話、ハッカソンなどを実施しました。詳しくは以下のredmineの過去のチケットを参照してください。<br>
https://rm.lsnl.jp/projects/workshop/issues<br>
- 発表者が喋って聴講者が聞くという講義形式ではなく(それでもいいですが)、できればできるだけ参加者がコーディングしたり、質問や意見を交換しながらインタラクティブに学ぶことを目指しています。<br>
### 勉強会の目的
- 学びのきっかけを広げる<br>
 各自がそれぞれ興味のある分野について発表することで、各々が知見を広げる機会を得る。<br>
- 効率の良い学習を実現する<br>
 学習コストと学習利益の観点から、各自が学習した内容を互いに発表し合うと1人分の学習コストから参加人数分の学習利益を得られる。これを目指す。<br>
### 勉強会のルール(適宜更新が必要?)
- 全員が発表の機会を持つ<br>
 発表者の増加はそのまま学習効率の向上に繋がるため、極力全員が発表を行えるようにする。<br>
 効率の良い学習を実現するために、少し硬い表現になりますが、下記のredmineのwikiを見ていただければと思います。<br>
https://rm.lsnl.jp/projects/adm/wiki/%E5%8B%89%E5%BC%B7%E4%BC%9A%E8%BC%AA%E8%AC%9B%E3%81%AE%E5%8E%9F%E7%90%86<br>
要約すると、**give and take を意識することで、学習効率を{参加人数}倍にする**ことを目標にしています。<br>
例えば参加人数が5人だと、発表を一人一回ずつ行うと学習効率は理想的には5倍になる。<br>
ただし、それは各々が同じ量を学習し、その内容を発表した場合に限られる<br>
普通にやると、人間は弱いので、"自分は聴講だけしたい"とか、"発表の準備をしてない"(~=フリーライダー？)となり、真面目に発表した人が損をする(真面目に発表した人は自分が発表することで一人で学習する5倍の量の知識を得られると期待しているため)。<br>
学習効率が{参加人数}倍に近づけられるように、参加者全員が自覚をもって取り組むことが重要<br>
- 時間厳守<br>
研究室で何かイベントをすると、時間を押してしまうことが多いのですが、勉強会は時間厳守で行きたいです。<br>
１コマ(100分)丸々やるのではなくて、60分ぐらいで終わる感じで行きたいなと思っています(ボリュームが少なすぎてもダメですが、多すぎてもダメです)。<br>
- 勉強会のリマインド<br>
開催日の前日(月曜日)にリマインドのメールをall@lsnl.jpとtraining-a@lsnl.jp宛てに送ってください。<br>
メールには、勉強会の構成(何をやるか)を簡単に書いてください。<br>
- 資料は無くてもいい<br>
勉強会に使う資料は無くてもいいですが、記録を残すという観点でも、理解度促進という観点でも、できればあった方が良いので作ってもらえると嬉しいです。<br>
- できれば対面で参加してほしい<br>
上記でも述べましたが、参加者同士がインタラクティブに進行していきたいので、対面参加をお願いしたいですが、Zoomでも参加できるように画面共有をしながらやる予定です。<br>
### 勉強会の開催頻度
現時点では、毎週火曜日３限にしようと思っています。<br>
二週間に一回がいいとか、月に一回がいい等の意見が多ければ調整しようと思っています<br>
ただし、論文提出の直前や夏季休暇中、春期休暇中、祝日は実施しません(今年のソサイエティ大会の締め切りが6/27なので(変わるかも？)、6月は実施しないかも)。<br>
ソサイエティ大会2025:https://www.ieice.org/jpn_r/activities/taikai/society/2025/
### 勉強会の担当者(発表者)
発表した人が、次に発表する人を指名する形式でやろうかなと思っています。(例: kazuma-aが発表した場合、勉強会が終わった時に、次の発表者としてyoshiteruを指名する。)<br>
と思っていましたが、予め発表者を決めて置いた方が準備がしやすい？<br>
### 勉強会係(Hidehiro, Tetsuro)にやってほしいこと
- 勉強会の司会<br>
始めのあいさつとかタイムキーパーとか？<br>
- リマインドメールの確認<br>
開催日の前日(月曜日)までにリマインドのメールが来ていない場合、発表者に送るように伝える<br>
- 勉強会の内容を何かしらで保存する<br>
勉強会をやって終わりだともったいないので、内容をどこかに保存するとか、どこかに公開するとか何かしらしてほしい。(これは広報係のタスク？)<br>
- 参加者が増えるように策を打つ<br>
参加率が100%になるように何か策を打ってほしい<br>

## 質疑応答

## 2.The Annotated Transformer①(導入+環境構築)
### "The Annotated Transformer"とは?
### "Attention is All You Need"とは?
### pythonの環境構築(仮想環境推奨)
### 必要なmodule一覧

```
a = 10
print("hello")
```
```python
a = 10
print("hello")
```

## 参考文献
[1] Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman(2022). The Annotated Transformer. https://nlp.seas.harvard.edu/annotated-transformer/<br>
[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser(2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30 (NIPS 2017)<br>
[3] 森下篤(2024). Visual Studio Code 実践ガイド. 技術評論社<br>
[4] Bill Ludanovic, 鈴木駿, 長尾高弘(2022). 入門 Python3 第二版. O'Reilly Japan<br>
[5] Al Sweigart, 相川愛三(2023). 退屈なことはPythonにやらせよう　第二版. O'Reilly Japan<br>
[6] Al Sweigart, 岡田祐一(2022). きれいなPythonプログラミング. マイナビ<br>

## 更新履歴
2025.04.09 first commit<br>
2025.04.09 update readme(add structure of readme)<br>
2025.04.09 fix format of reference<br>
2025.04.09 add program folder and update readme<br>
2025.04.10 update readme<br>
2025.04.10 refrect the opinion of yoshiteru<br>


Kazuma Aoyama(kazuma-a@lanl.jp), Yoshiteru Taira(yohsiteru@lsnl.jp)
