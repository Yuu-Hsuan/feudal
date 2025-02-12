
# 引入函式庫
```
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected, flatten, conv2d
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.distributions import Categorical
```
* `numpy`：用於數值運算。
* `tensorflow`：主要的深度學習框架。
* `tensorflow.contrib.layers`：
  * `fully_connected`：全連接層。
  * `flatten`：將多維數據攤平成一維。
  * `conv2d`：2D 卷積層。
* `tensorflow.contrib.rnn.BasicLSTMCell`：基本的 LSTM 單元。
* `tensorflow.contrib.distributions.Categorical`：用於抽樣動作。

# 引入 SC2 相關函式庫
```
from pysc2.lib import actions
from pysc2.lib import features

from rl.common.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
from rl.common.util import mask_unavailable_actions
from rl.networks.util.cells import ConvLSTMCell
```
* `pysc2.lib.actions`：SC2 動作庫。
* `pysc2.lib.features`：SC2 狀態特徵庫。
* `is_spatial_action`：判斷動作是否與空間相關。
* `NUM_FUNCTIONS`：動作數量。
* `FLAT_FEATURES`：非空間特徵。
* `mask_unavailable_actions`：過濾不可用的動作。
* `ConvLSTMCell`：卷積 LSTM 單元

# Feudal 類別
這個類別定義了一個 Feudal Network，並實作其輸入處理、嵌入 (embedding)、LSTM 狀態管理及行動策略。
```
class Feudal:
    """Feudal Networks network implementation based on https://arxiv.org/pdf/1703.01161.pdf"""

    def __init__(self, sess, ob_space, nbatch, nsteps, d, k, c, reuse=False, data_format='NCHW'):
        # sess：TensorFlow 會話 (session)，用來執行計算圖 (graph)
        # ob_space：觀察空間 (包含 screen、minimap、flat 等)。代表環境提供的數據（遊戲畫面、地圖資訊等）。
        # nbatch：批次大小。每個訓練步驟處理的樣本數
        # nsteps：時間步長。表示模型一次訓練會看多少個時間點的數據
        # d：管理層 (manager) 的目標向量維度。高層管理者的隱藏層維度。
        # k：特徵提取的維度。工人的隱藏層維度。
        # c：通道數。管理者的記憶儲存大小（決定能記住多少步的資訊）
        # reuse：是否重用變數。是否重複使用模型（在測試時可用）
        # data_format：數據格式 (NCHW 或 NHWC)，但目前僅支援 NHWC。資料格式，NCHW（批次、通道、高度、寬度）或 NHWC（批次、高度、寬度、通道）。
        # BUG: does not work with NCHW yet.
        if data_format=='NCHW':
            print('WARNING! NCHW not yet implemented for ConvLSTM. Switching to NHWC')
        data_format='NHWC'
#------------------------------------------------------------------------------------------------------
# 觀察特徵嵌入 (Embedding)
        def embed_obs(x, spec, embed_fn, name):
            # embed_obs：處理 screen、minimap 和 flat 特徵，將類別型特徵 (categorical) 轉換為 One-hot，將數值型特徵 (scalar) 取對數

            # 根據 spec（特徵規格）來分割 x（輸入資料），並沿著最後一個維度 (-1) 進行切分。例如：如果 x 有 3 個通道，那這行程式會把它拆成 3 個獨立的變數。
            feats = tf.split(x, len(spec), -1)
            
            out_list = []

            # 對每個特徵進行轉換，這行表示我們會 逐一處理每個特徵
            for s in spec:
                f = feats[s.index]

                # 處理類別特徵（Categorical Features）
                ## s.scale 代表這個特徵有多少種可能的值（例如，假設是 256）
                ## log2(s.scale) 計算出適當的編碼維度，例如 log2(256) = 8，這樣可以用 8 維向量來表示這個特徵
                ## one_hot 把類別值轉換成 one-hot 編碼，例如：3 轉換成 [0, 0, 0, 1, 0, 0, 0, 0]
                ## embed_fn 是一個神經網路層，它可以把 one-hot 向量壓縮成更小的維度
                if s.type == features.FeatureType.CATEGORICAL:
                    dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                    dims = max(dims, 1)
                    indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
                    out = embed_fn(indices, dims, "{}/{}".format(name,s.name))
                elif s.type == features.FeatureType.SCALAR:
                    out = tf.log(f + 1.)
                else:
                    raise NotImplementedError
                out_list.append(out)
            return tf.concat(out_list, -1)
#------------------------------------------------------------------------------------------------------
# 空間與非空間特徵嵌入

        def embed_spatial(x, dims, name):
            # embed_spatial：對空間特徵做 1x1 卷積嵌入
            x = from_nhwc(x)
            out = conv2d(
                x, dims,
                kernel_size=1,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv_embSpatial" % name)
            return to_nhwc(out)


        def embed_flat(x, dims, name):
            # embed_flat：對非空間特徵做全連接嵌入
            return fully_connected(
                x, dims,
                activation_fn=tf.nn.relu,
                scope="%s/conv_embFlat" % name)
#------------------------------------------------------------------------------------------------------
# 深度學習影像處理函數:適用於 CNN (卷積神經網路)，並針對空間特徵 (spatial features) 和非空間特徵 (non-spatial features) 進行處理。
        #-----------------------------------------------------------------------------------------------------
        # 功能： 兩層 卷積層 (Conv2D)，用來提取輸入影像的特徵
        def input_conv(x, name):
            ## x：輸入影像 (可能是 NHWC 或 NCHW 格式)
            ## name：用來給這層命名，方便之後識別

            # 第一層卷積 (5x5 Kernel, 16 個通道)
            ## 作用： 這一層會對輸入影像執行 5x5 的卷積，輸出 16 個特徵圖，並使用 ReLU 讓結果非線性化。
            conv1 = conv2d(
                x, 16,  # 16 個輸出通道 (Filters)
                kernel_size=5,  # 卷積核大小為 5x5
                stride=1,  # 步長 = 1 (不縮小影像大小)
                padding='SAME',  # 保持輸出大小不變
                activation_fn=tf.nn.relu,  # ReLU 激活函數
                data_format=data_format,  # 資料格式 (NHWC 或 NCHW)
                scope="%s/conv1" % name)  # 命名這層為 "name/conv1"

            # 第二層卷積 (3x3 Kernel, 32 個通道)
            ## 這一層使用 3x3 卷積，讓特徵圖從 16 通道變成 32 通道，進一步提取影像特徵
            conv2 = conv2d(
                conv1, 32,  # 32 個輸出通道
                kernel_size=3,  # 卷積核大小為 3x3
                stride=1,  # 步長 = 1
                padding='SAME',  # 保持輸出大小不變
                activation_fn=tf.nn.relu,  # ReLU 激活函數
                data_format=data_format,  # 資料格式
                scope="%s/conv2" % name)  # 命名這層為 "name/conv2"
            return conv2   # 返回經過兩層卷積處理的影像特徵圖

        #-----------------------------------------------------------------------------------------------------
        # 功能： 產生 非空間輸出 (Non-Spatial Output)，通常適用於分類問題
        def non_spatial_output(x, channels, name):
            ## x：輸入特徵向量 (通常是全連接層的輸出)
            ## channels：輸出的維度 (通常等於分類數量)
            ## name：名稱

            ## 全連接層 (Fully Connected Layer)
            ### 這裡使用一個 全連接層 (FC Layer)，將輸入 x 轉換成 channels 維的向量，但 不使用激活函數，讓原始 logits 直接輸出
            logits = fully_connected(x, channels, activation_fn=None, scope="non_spatial_output/flat/{}".format(name))
            ### 這裡應用 Softmax 函數，將 logits 轉換為機率分佈，適用於分類問題
            return tf.nn.softmax(logits)

        #-----------------------------------------------------------------------------------------------------
        # 功能： 產生 空間輸出 (Spatial Output)，通常用於像素級分類 (如語意分割、對應策略圖等)
        ## x：輸入的影像特徵圖
        ## name：名稱
       def spatial_output(x, name):

            # 這裡使用 1x1 卷積，將輸入 x 降維為單通道輸出 (代表每個像素的分數)
            logits = conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None, data_format=data_format, scope="spatial_output/conv/{}".format(name))

            # 格式轉換與展平
            ## to_nhwc(logits)：確保輸出是 NHWC 格式
            ## flatten(logits)：將影像展平成向量，方便計算機率
            logits = flatten(to_nhwc(logits), scope="spatial_output/flat/{}".format(name))

            # 應用 Softmax，將結果轉換為機率分佈，適合用來判斷每個像素的分類
            return tf.nn.softmax(logits)

        #-----------------------------------------------------------------------------------------------------
        # 功能： 在 通道 (Channel) 維度上合併多個 2D 特徵圖
        ## lst：要合併的特徵圖列表
        def concat2DAlongChannel(lst):
            """Concat along the channel axis"""
            # 選擇合併的軸:若 data_format == 'NCHW'，則通道軸是第 1 維，若 data_format == 'NHWC'，則通道軸是第 3 維
            axis = 1 if data_format == 'NCHW' else 3
            # 合併特徵圖:將多個特徵圖沿著通道軸合併，形成一個更豐富的特徵表示
            return tf.concat(lst, axis=axis)
        #-----------------------------------------------------------------------------------------------------
        # 功能： 將 1D 向量展開成 2D 影像大小，用來廣播 非空間特徵 到整個影像區域
        ## flat：1D 特徵向量
        ## size2d：影像的 (高度, 寬度)
        def broadcast_along_channels(flat, size2d):

            # 展開維度:先 增加兩個維度 (expand_dims)，再透過 tile 複製特徵向量到整個 2D 空間。
            if data_format == 'NCHW':
                return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                               tf.stack([1, 1, size2d[0], size2d[1]]))
            return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                           tf.stack([1, size2d[0], size2d[1], 1]))
        #-----------------------------------------------------------------------------------------------------
        # 功能： 格式轉換 (NHWC ↔ NCHW)
        ## NCHW → NHWC：將 (batch, channels, height, width) 轉換為 (batch, height, width, channels)
        def to_nhwc(map2d):
            if data_format == 'NCHW':
                return tf.transpose(map2d, [0, 2, 3, 1])
            return map2d

        def from_nhwc(map2d):
            if data_format == 'NCHW':
                return tf.transpose(map2d, [0, 3, 1, 2])
            return map2d

        nenvs = nbatch//nsteps
        res = ob_space['screen'][1]
        filters = k
        ncores = c
        m_state_shape = (2, nenvs, ncores, d)
        w_state_shape = (2, nenvs, res, res, filters)
        lc_shape = (None, c, d)

        SCREEN  = tf.placeholder(tf.float32, shape=ob_space['screen'],  name='input_screen')
        MINIMAP = tf.placeholder(tf.float32, shape=ob_space['minimap'], name='input_minimap')
        FLAT    = tf.placeholder(tf.float32, shape=ob_space['flat'],    name='input_flat')
        AV_ACTS = tf.placeholder(tf.float32, shape=ob_space['available_actions'], name='available_actions')

        STATES = {
            'manager' : tf.placeholder(tf.float32, shape=m_state_shape, name='initial_state_m'),
            'worker'  : tf.placeholder(tf.float32, shape=w_state_shape, name='initial_state_w')
        }
        LAST_C_GOALS = tf.placeholder(tf.float32, shape=lc_shape, name='last_c_goals')
        LC_MANAGER_OUTPUTS = tf.placeholder(tf.float32, shape=lc_shape, name='lc_manager_outputs')
#------------------------------------------------------------------------------------------------------

        with tf.variable_scope('model', reuse=reuse):

            screen_emb  = embed_obs(SCREEN,  features.SCREEN_FEATURES,  embed_spatial, 'screen')
            minimap_emb = embed_obs(MINIMAP, features.MINIMAP_FEATURES, embed_spatial, 'minimap')
            flat_emb    = embed_obs(FLAT, FLAT_FEATURES, embed_flat, 'flat')

            screen_out    = input_conv(from_nhwc(screen_emb), 'screen')
            minimap_out   = input_conv(from_nhwc(minimap_emb), 'minimap')
            broadcast_out = broadcast_along_channels(flat_emb, ob_space['screen'][1:3])
            print('flat', flat_emb)
            print('flat bc', broadcast_out)

            z = concat2DAlongChannel([screen_out, minimap_out, broadcast_out])
            flattened_z = flatten(z)

# 經理 (Manager) 與工人 (Worker) 模組
            # 經理模組
            with tf.variable_scope('manager'):
                # REVIEW: maybe we want to put some strided convolutions in here because flattening
                # z gives a pretty big vector.
                # Dimensionaliy reduction on z to get R^d vector.

                s = fully_connected(flattened_z, d, activation_fn=tf.nn.relu, scope="s")
                # s：將 flattened_z 投影到 d 維度。
                #print('s', s, s.shape)

                manager_LSTM_input = tf.reshape(s, shape=(nenvs,nsteps,d))
                # manager_LSTM_input：調整維度以符合 LSTM 輸入格式
                #print('manager_LSTM_input', manager_LSTM_input, manager_LSTM_input.shape)

                manager_cell = BasicLSTMCell(d, activation=tf.nn.relu)
                # manager_cell：LSTM 單元，負責管理層目標
                print("state input shape: ", STATES['manager'][0,:,0,:])

                # g_：LSTM 輸出 (目標向量)
                # h_M：LSTM 狀態
                g_, h_M = tf.nn.dynamic_rnn(
                    manager_cell,
                    manager_LSTM_input,
                    initial_state=tf.nn.rnn_cell.LSTMStateTuple(STATES['manager'][0,:,0,:], STATES['manager'][1,:,0,:]),
                    time_major=False,
                    dtype=tf.float32,
                    scope="manager_lstm"
                )

                print(tf.expand_dims(h_M, axis=2))
                #print("g_", g_)
                dilated_state = tf.concat([STATES['manager'][:,:,1:,:], tf.expand_dims(h_M, axis=2)], axis=2)
                #print("dilated state", dilated_state)
                dilated_outs = tf.concat([LC_MANAGER_OUTPUTS[:, 1:, :], tf.reshape(g_, (nenvs*nsteps, 1, d))], axis = 1)

                #print("dilated outs ", dilated_outs)
                g_hat = tf.reduce_sum(dilated_outs, axis=1)
                epsilon = 0.001 #TODO: add some sort of decay here based on global_step (polynomial?)
                goal = tf.cond(tf.random_uniform([1], minval=0, maxval=1)[-1] < epsilon, lambda: tf.nn.l2_normalize(tf.random_normal((nenvs*nsteps, d), mean=0, stddev=1), dim=1), lambda: tf.nn.l2_normalize(g_hat, dim=1))
                # Manger Value
                manager_value_fc = fully_connected(flattened_z, 256, activation_fn=tf.nn.relu)
                manager_value = fully_connected(manager_value_fc, 1, activation_fn=None, scope="manager_value")
                print("manager_value ", manager_value)
                manager_value = tf.reshape(manager_value, [-1])

            # 工人模組
            with tf.variable_scope('worker'):

                # TODO: make (dilated) ConvLSTM Cell
                # TODO: what's the dimensions in batch_size??
                convLSTM_shape = tf.concat([[nenvs, nsteps],tf.shape(z)[1:]], axis=0)
                convLSTM_inputs = tf.reshape(z, convLSTM_shape)
                convLSTMCell = ConvLSTMCell(shape=ob_space['screen'][1:3], filters=filters, kernel=[3, 3], reuse=reuse) # TODO: padding: same?
                # convLSTMCell：工人層使用 ConvLSTMCell，適合空間資訊處理。

                # convLSTM_outputs：LSTM 輸出
                # convLSTM_state：LSTM 狀態
                convLSTM_outputs, convLSTM_state = tf.nn.dynamic_rnn(
                    convLSTMCell,
                    convLSTM_inputs,
                    initial_state=tf.nn.rnn_cell.LSTMStateTuple(STATES['worker'][0], STATES['worker'][1]),
                    time_major=False,
                    dtype=tf.float32,
                    scope="worker_lstm"
                )
                print(convLSTM_state)
                # TODO: what's the dimensions in batch_size??

                U = tf.reshape(convLSTM_outputs, tf.concat([[nenvs*nsteps],tf.shape(convLSTM_outputs)[2:]], axis=0))

                cut_g = tf.stop_gradient(goal)
                cut_g = tf.expand_dims(cut_g, axis=1)
                g_stack = tf.concat([LAST_C_GOALS, cut_g], axis=1)
                last_c_g = g_stack[:,1:,:]
                g_sum = tf.reduce_sum(last_c_g, axis=1)

                phi = tf.get_variable("phi", shape=(d, k))
                w = tf.matmul(g_sum, phi)
                U_w = tf.multiply(U, tf.reshape(w, (nenvs*nsteps, 1, 1, k)))

                flat_out = flatten(U_w, scope='flat_out')
                fc = fully_connected(flat_out, 256, activation_fn=tf.nn.relu, scope='fully_con')

                worker_value_fc = fully_connected(flattened_z, 256, activation_fn=tf.nn.relu)
                worker_value = fully_connected(worker_value_fc, 1, activation_fn=None, scope='value')
                worker_value = tf.reshape(worker_value, [-1])
                print("worker_value ", worker_value)
#------------------------------------------------------------------------------------------------------
# 動作與價值
                fn_out = non_spatial_output(fc, NUM_FUNCTIONS, 'fn_name')

                args_out = dict()
                for arg_type in actions.TYPES:
                    if is_spatial_action[arg_type]:
                        arg_out = spatial_output(U_w, name=arg_type.name)
                    else:
                        arg_out = non_spatial_output(fc, arg_type.sizes[0], name=arg_type.name)
                    args_out[arg_type] = arg_out

                policy = (fn_out, args_out)
                # policy：包含函數名稱 (fn_out) 及動作參數 (args_out)。

                value = (manager_value, worker_value)
                # value：包含 manager_value 和 worker_value。
#------------------------------------------------------------------------------------------------------
        def sample_action(available_actions, policy):

            def sample(probs):
                dist = Categorical(probs=probs, allow_nan_stats=False)
                return dist.sample()

            fn_pi, arg_pis = policy
            fn_pi = mask_unavailable_actions(available_actions, fn_pi)
            fn_samples = sample(fn_pi)

            arg_samples = dict()
            for arg_type, arg_pi in arg_pis.items():
                arg_samples[arg_type] = sample(arg_pi)

            return fn_samples, arg_samples

        action = sample_action(AV_ACTS, policy)

        #TODO:recheck inputs of feed_dicts

        def step(obs, state, last_goals, m_out, maks=None):
            """
            Receives observations, hidden states and goals at a specific timestep
            and returns actions, values, new hidden states and goals.
            """
            feed_dict = {
                SCREEN           : obs['screen'],
                MINIMAP          : obs['minimap'],
                FLAT             : obs['flat'],
                AV_ACTS          : obs['available_actions'],
                LAST_C_GOALS     : last_goals,
                STATES['manager']: state['manager'],
                STATES['worker'] : state['worker'],
                LC_MANAGER_OUTPUTS: m_out
            }
            a, v, _h_M, _h_W, _s, g, m_o = sess.run([action, value, dilated_state, convLSTM_state, s, last_c_g, dilated_outs], feed_dict=feed_dict)
            state = {'manager':_h_M,'worker':_h_W}
            return a, v, state, _s, g, m_o


        def get_value(obs, state, last_goals, m_out, mask=None):
            """
            Returns a tuple of manager and worker value.
            """
            feed_dict = {
                SCREEN           : obs['screen'],
                MINIMAP          : obs['minimap'],
                FLAT             : obs['flat'],
                LAST_C_GOALS     : last_goals,
                STATES['manager']: state['manager'],
                STATES['worker'] : state['worker'],
                LC_MANAGER_OUTPUTS: m_out
            }
            return sess.run(value, feed_dict=feed_dict)


        self.SCREEN  = SCREEN
        self.MINIMAP = MINIMAP
        self.FLAT    = FLAT
        self.AV_ACTS = AV_ACTS
        self.LAST_C_GOALS = LAST_C_GOALS
        self.STATES = STATES
        self.goal = goal
        self.LC_MANAGER_OUTPUTS = LC_MANAGER_OUTPUTS

        self.z = flattened_z
        self.s = s
        self.w = w
        self.u = U

        self.policy = policy
        self.step = step
        self.value = value
        self.get_value = get_value
        self.initial_state = {
            'manager' : np.zeros(m_state_shape, dtype=np.float32),
            'worker'  : np.zeros(w_state_shape, dtype=np.float32)
        }
```
