import numpy as np

'''
テクニカル指標の関数たち

計算式は，次を参考にしています
[MONEX Inc.](https://info.monex.co.jp/technical-analysis/indicators/)
[matplotlib/mplfinance](https://github.com/matplotlib/mplfinance)
'''

'''
ボリンジャーバンド

±1σ ＝ n日の移動平均 ± n日の標準偏差
±2σ ＝ n日の移動平均 ± n日の標準偏差 × 2
±3σ ＝ n日の移動平均 ± n日の標準偏差 × 3

＜ポイント＞価格がバンド内に収まる確率について
ボリンジャーバンドの±1σの範囲内に収まる確率　⇒　約68.3％
ボリンジャーバンドの±2σの範囲内に収まる確率　⇒　約95.4％
ボリンジャーバンドの±3σの範囲内に収まる確率　⇒　約99.7％    
[MONEX, Inc.](https://info.monex.co.jp/technical-analysis/indicators/003.html)
'''
def bollingerband(c, period):
    bbma = c.rolling(window=period).mean() ## 平均
    bbstd = c.rolling(window=period).std() ## 標準偏差
    bbh1 = bbma + bbstd * 1
    bbl1 = bbma - bbstd * 1
    bbh2 = bbma + bbstd * 2
    bbl2 = bbma - bbstd * 2
    bbh3 = bbma + bbstd * 3
    bbl3 = bbma - bbstd * 3
    return bbh1, bbl1, bbh2, bbl2, bbh3, bbl3

'''
MACD

MACD＝短期EMA－長期EMA
MACDシグナル＝MACDのEMA

※MACDに用いられる移動平均は「単純移動平均（SMA)」ではなく、「指数平滑移動平均（EMA）」を使います。
  EMAは直近の値動きをより反映するため、SMAと比較して値動きに敏感に反応すると考えらます。

ポイント
パラメータ値は、短期EMAが12、長期EMAが26、MACDシグナルが9に設定する場合が多いです。
※ただし、銘柄ごとやマーケット状況に応じてパラメータ値の変更が必要になります。

MACDとMACDシグナルのゴールデンクロスで買い、デッドクロスで売り。
[MONEX, Inc.](https://info.monex.co.jp/technical-analysis/indicators/002.html)
[matplotlib/mplfinance](https://github.com/matplotlib/mplfinance)

adjust=False については
[pandas.Series.ewm](https://pandas.pydata.org/docs/reference/api/pandas.Series.ewm.html)
'''
def macd(c, n1, n2, ns):
    ema_short = c.ewm(span=n1,adjust=False).mean()
    ema_long = c.ewm(span=n2,adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=ns,adjust=False).mean()
    histogram = macd - signal
    histogramplus = histogram.where(histogram > 0, 0)
    histogramminus = histogram.where(histogram < 0, 0)
    return macd, signal, histogram, histogramplus, histogramminus

'''
RSI（指数平滑移動平均版）

① RS＝（n日間の終値の上昇幅の平均）÷（n日間の終値の下落幅の平均）
② RSI= 100-（100÷（RS+1））
ポイント
n(パラメータ値)は考案者であるJ.W.ワイルダー氏が最適とする“14”（日足）と設定する場合が多いです。
他パラメータ値としては、日足では9日、22日、42日、52日。週足では9週、13週です。
※期間設定は、もっとも効果がでると判断できるものを使えばそれが正解になりますので、既成概念にとらわれる必要はありません。

[MONEX, Inc.](https://info.monex.co.jp/technical-analysis/indicators/005.html)
'''
def rsi(c, period):
    diff = c.diff() #前日比
    up = diff.copy() #上昇
    down = diff.copy() #下落
    up = up.where(up > 0, np.nan) #上昇以外はnp.nan
    down = down.where(down < 0, np.nan) #下落以外はnp.nan
    #upma = up.rolling(window=period).mean() #平均
    #downma = down.abs().rolling(window=period).mean() #絶対値の平均
    upma = up.ewm(span=period,adjust=False).mean() #平均
    downma = down.abs().ewm(span=period,adjust=False).mean() #絶対値の平均
    rs = upma / downma
    rsi = 100 - (100 / (1.0 + rs))
    return rsi

'''
一目均衡表

- 基準線=（当日を含めた過去26日間の最高値+最安値）÷2
- 転換線=（当日を含めた過去9日間の最高値+最安値）÷2
- 先行スパン1=｛（転換値+基準値）÷2｝を26日先行させて表示
- 先行スパン2=｛（当日を含めた過去52日間の最高値+最安値）÷2｝を26日先行させて表示
- 遅行スパン= 当日の終値を26日遅行させて表示
[MONEX, Inc.](https://info.monex.co.jp/technical-analysis/indicators/004.html)
'''
def ichimoku(o, h, l, c):
    ## 当日を含めた過去26日間の最高値
    ## 当日を含めた過去 9日間の最高値
    ## 当日を含めた過去52日間の最高値
    max26 = h.rolling(window=26).max()
    max9  = h.rolling(window=9).max()
    max52 = h.rolling(window=52).max()
    ## 当日を含めた過去26日間の最安値
    ## 当日を含めた過去 9日間の最安値
    ## 当日を含めた過去52日間の最安値
    min26 = l.rolling(window=26).min()
    min9  = l.rolling(window=9).min()
    min52 = l.rolling(window=52).min()

    ## 基準線=（当日を含めた過去26日間の最高値+最安値）÷2
    ## 転換線=（当日を含めた過去9日間の最高値+最安値）÷2
    kijun = (max26 + min26) / 2
    tenkan = (max9 + min9) / 2
    ## 先行スパン1=｛（転換値+基準値）÷2｝を26日先行させて表示
    senkospan1 = (kijun + tenkan) / 2
    senkospan1 = senkospan1.shift(26)
    ## 先行スパン2=｛（当日を含めた過去52日間の最高値+最安値）÷2｝を26日先行させて表示
    senkospan2 = (max52 + min52) / 2
    senkospan2 = senkospan2.shift(26)
    ## 遅行スパン= 当日の終値を26日遅行させて表示
    chikouspan = c.shift(-26)

    return kijun, tenkan, senkospan1, senkospan2, chikouspan
