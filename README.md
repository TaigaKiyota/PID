# このディレクトリの使い方

主要スクリプトはノイズを含むハミルトン系の制御を扱います。以下の順で使うと結果の再現や再実験が楽です。

## 0. 依存関係の準備
- 初回のみ `julia Pkg_download.jl` を実行して必要パッケージを入れます。

## 1. システム設定ファイル（Settings.jld2）の作成
- `GainSettingMaiking_Noise.jl` を実行すると `System_setting/Noise_dynamics/Settings/Setting<番号>/Settings.jld2` が生成されます。
- 変えたい場所: ファイル先頭の `Setting_num`（使用する設定の番号）、各 `elseif` 節内の次元・ノイズ共分散・目標値など。

## 2. 実験セットの下ごしらえ（params.json などの作成）
- 共通で `Setting_num` と `simulation_name` を揃えると、同じディレクトリ `System_setting/Noise_dynamics/Settings/Setting<番号>/VS_ModelBase/<simulation_name>` を共有できます。
- まず初めに`Comparison_vsModelBase_zoh.jl`でパラメータを設定して，'trial1'が出るまで実行させるとパラメータを保存したフォルダが作成される．
- 主なスクリプトと役割:
  - `Comparison_vsModelBase_zoh.jl`: システム同定→フィードフォワード推定→モデルフリー/モデルベースのゲイン探索をまとめて実行。`params.json` や各種リスト (`list_uhat.jld2`, `list_Kp_seq_*`, `list_Ki_seq_*` など) を保存。
  - `FF_vsModelBase_zoh.jl`: フィードフォワード推定のみ比較したいときの軽量版。`params.json`, `list_uhat.jld2`, `list_ustar_Sysid.jld2` を保存。
  - 複数設定をまとめて回す場合は `Comparison_SomeSetting_zoh.jl` / `Comparison_SomeSetting_Identification_zoh.jl`（`dir_experiment_setting` でフォルダ指定）を使います。
- 変えたい場所: 各ファイル冒頭の `Setting_num`, `simulation_name`、アルゴリズムパラメータ（`eta`, `epsilon_GD`, `N_sample`, `N_GD`, `tau`, `r` など）、同定条件（`Ts`, `Num_Samples_per_traj`, `PE_power`, `accuracy` など）。

## 3. システム同定だけやり直したいとき
- `Comparison_Identification.jl` `Comparison_SomeSetting_Identification.jl`を実行すると `params.json` で指定した条件で同定を繰り返し、`list_est_system.jld2` を生成・更新します。
- 変えたい場所: 冒頭の `Setting_num`, `simulation_name`、`params.json` 側の `Ts`, `Num_Samples_per_traj`, `PE_power`, `accuracy`。

## 4. ゲイン最適化を個別に回す
- モデルフリー最適化: `ModelFree_Gradient.jl` `ModelFree_SomeSetting_Gradient.jl`。`params.json` を読み込み、`Trials` 回の勾配推定を実施し `list_Kp_seq_ModelFree.jld2`, `list_Ki_seq_ModelFree.jld2`, `list_uhat.jld2` を更新します。主に変えるのは冒頭の `Setting_num`, `simulation_name`, `eta`, `epsilon_GD`, `N_GD`, `tau`, `r`, `K_P_uhat`。
- モデルベース最適化: `ModelBased_Gradient.jl`。`list_est_system.jld2` を入力に `list_Kp_Sysid.jld2`, `list_Ki_Sysid.jld2` を生成します。変える場所は `Setting_num`, `simulation_name`, `eta_discrete`, `epsilon_GD_discrete`, `N_GD_discrete`, `projection` など。
- どちらも `Problem_param` の `Q1`, `Q2`（重み行列）を先頭近くで設定しているので、評価基準を変えたい場合はここを編集します。

## 5. 結果の可視化・評価
- フィードフォワード・ゲインの誤差箱ひげ図や軌道図: `Visualization_VSModelBase.jl`（連続系）と `Zoh_Gain_Visualization_VSModelBase.jld`（ZOH 版）。`params.json` と各種 `list_*.jld2` を読み込んで図を `VS_ModelBase/<simulation_name>/` 以下に保存します。
- 目的関数値の比較: `Comparison_ObjFunc.jl` が `list_Kp_seq_*` / `list_Ki_seq_*` を使ってモデルフリー制御器とモデルベースで計算された ZOH 制御器の平均目的関数を計算し、箱ひげ図を保存します。

## メモ
- ほとんどのスクリプトは乱数シードを固定しているので、条件変更は冒頭のパラメータを触るだけで十分です。
- `simulation_name` ごとに出力フォルダが分かれるので、複数条件を試すときは名前を変えるか別フォルダを作ってください。
