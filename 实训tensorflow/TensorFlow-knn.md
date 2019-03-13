1.导入数据集

```Python
from sklearn.datasets import load_iris
iris = load_iris()
```

2.将数据集按8:2划分

```Python
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2)
```

3.占位符

```
train = tf.placeholder(dtype=tf.float32, shape=[None,4])
test = tf.placeholder(dtype=tf.float32, shape=[4])
#超参数k
k = tf.placeholder(dtype=tf.int32, shape=[])
```

4.L1距离：

```Python
dist = tf.reduce_sum(tf.abs(tf.add(train,tf.negative(test))),
                     reduction_indices=1)
```

5.某个样本点的最小距离的索引：

```python
#tf.nn.top_k返回最大的几个数
value,index= tf.nn.top_k(-dist,k=k)
```

6.会话：

```Python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(1,10):
        acc = 0
        for i in range(x_test.shape[0]):
            id = sess.run(index,
                          feed_dict={train:x _train,
                                     test:x_test[i,:],
                                     k:j})
            labels = y_train[id]
            predict = np.argmax(np.bincount(labels))
            true = y_test[i]
            if predict == true:
                acc = acc+1
        print(acc/len(x_test))
```

7.总结：

当k = 5时，其正确率最大，为96.67%。