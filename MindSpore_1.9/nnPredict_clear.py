import mindspore
from mindspore import context, nn, ops
#from src.nn import Network     ## 明文下的神经网络模型
from dist.nn import Network   ## pyarmor加密后的 神经网络模型

context.set_context(mode=mindspore.PYNATIVE_MODE) #设置为动态图
model = Network()
print(model)

X = ops.ones((1, 28, 28), mindspore.float32)

logits = model(X)
print(logits)


pred_probab = nn.Softmax(axis=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
