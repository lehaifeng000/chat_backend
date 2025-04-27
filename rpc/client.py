# client.py
import Pyro5.api

# uri = "PYRO:model_default@127.0.0.1:50000"  # ✅ 手动指定 URI
uri = "PYRO:model_default@127.0.0.1:50000"  # ✅ 手动指定 URI
model_path = Pyro5.api.Proxy(uri)
print(model_path.eval("2+2=?", None))  # 调用远程方法
