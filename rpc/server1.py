# server.py
import Pyro5.api

@Pyro5.api.expose
class Hello:
    def say_hello(self, name):
        return f"Hello, {name}"

def main():
    daemon = Pyro5.api.Daemon(host="0.0.0.0", port=50000)  # ✅ 指定监听地址和端口
    uri = daemon.register(Hello, objectId="hello")         # ✅ 固定 objectId
    print(f"Server ready. URI: {uri}")
    daemon.requestLoop()

if __name__ == "__main__":
    main()
