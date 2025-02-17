## 方法 1：`tf.Print` 會在計算圖執行時輸出值
在 `loss` 計算過程中插入 `tf.Print` 來確保這些變數的值被輸出

這樣在 `TensorFlow` 執行 `loss` 運算時，會在標準輸出（console）中顯示變數的值

```
manager_loss = tf.Print(manager_loss, [manager_loss], message="manager_loss: ")
worker_loss = tf.Print(worker_loss, [worker_loss], message="worker_loss: ")
manager_value_loss = tf.Print(manager_value_loss, [manager_value_loss], message="manager_value_loss: ")
worker_value_loss = tf.Print(worker_value_loss, [worker_value_loss], message="worker_value_loss: ")
entropy = tf.Print(entropy, [entropy], message="entropy: ")

loss = manager_loss + worker_loss + value_loss_weight * manager_value_loss \
     + value_loss_weight * worker_value_loss - entropy_weight * entropy
```

## 方法 2：修改 `train` 函數，使用 `sess.run`
若要在 `train` 時輸出這些變數的數值，可以修改 `train` 函數，在 `sess.run` 時加入這些變數
這樣，每次訓練時都會印出這些變數的數值，方便觀察 `loss` 的變化

```
def train(obs, states, actions, returns, returns_intr, adv_m, adv_w, s_diff, goals, m_out, summary=False):
    feed_dict = {
        train_model.SCREEN : obs['screen'],
        train_model.MINIMAP: obs['minimap'],
        train_model.FLAT   : obs['flat'],
        train_model.AV_ACTS: obs['available_actions'],
        ACTIONS[0]         : actions[0],
        R                  : returns,
        RI                 : returns_intr,
        ADV_M              : adv_m,
        ADV_W              : adv_w,
        S_DIFF             : s_diff,
        train_model.LAST_C_GOALS : goals,
        train_model.STATES['manager']: states['manager'],
        train_model.STATES['worker']: states['worker'],
        train_model.LC_MANAGER_OUTPUTS: m_out
    }
    feed_dict.update({ v: actions[1][k] for k, v in ACTIONS[1].items() })

    agent_step = self.train_step
    self.train_step += 1

    if summary:
        _, _step, _loss, _manager_loss, _worker_loss, _manager_value_loss, _worker_value_loss, _entropy, _summary = sess.run(
            [train_op, global_step, loss, manager_loss, worker_loss, manager_value_loss, worker_value_loss, entropy, train_summary_op],
            feed_dict=feed_dict
        )
        print(f"Step: {_step}, Loss: {_loss}, Manager Loss: {_manager_loss}, Worker Loss: {_worker_loss}, Manager Value Loss: {_manager_value_loss}, Worker Value Loss: {_worker_value_loss}, Entropy: {_entropy}")
        return _step, _loss, _summary
    else:
        _train_op, _loss, _manager_loss, _worker_loss, _manager_value_loss, _worker_value_loss, _entropy = sess.run(
            [train_op, loss, manager_loss, worker_loss, manager_value_loss, worker_value_loss, entropy],
            feed_dict=feed_dict
        )
        print(f"Loss: {_loss}, Manager Loss: {_manager_loss}, Worker Loss: {_worker_loss}, Manager Value Loss: {_manager_value_loss}, Worker Value Loss: {_worker_value_loss}, Entropy: {_entropy}")
        return _train_op, _loss, None
```
