import Pyro5.api

from predict_math import gen_model, eval_question


model, tokenizer, image_processor = gen_model()

@Pyro5.api.expose
class ModelMath:
    def eval(self, text, img_path):
        # 推理
        output = eval_question(model, tokenizer, image_processor, img_path, text)
        return output


def main():
    daemon = Pyro5.api.Daemon(host="0.0.0.0", port=50001)  # ✅ 指定监听地址和端口
    uri = daemon.register(ModelMath, objectId="model_math")         # ✅ 固定 objectId
    print(f"Server ready. URI: {uri}")
    daemon.requestLoop()

if __name__ == "__main__":
    main()
