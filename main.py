import streamlit as st
from PIL import Image
import litellm
from litellm import completion
import base64
from io import BytesIO
import os

# 環境変数からAPIキーを取得
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OpenAI APIキーが設定されていません。")

# APIキーをlitellmに設定
litellm.api_key = openai_api_key

# モデル選択のためのセレクタを追加
#model_options = ["claude-3-haiku-20240307","gpt-4o-mini", "claude-3-5-sonnet-20240620", "gemini-pro-vision"]
#selected_model = st.selectbox("使用するモデルを選択してください", model_options)

# 機能選択のためのセレクタを追加
function_options = ["ボケて", "褒めて", "ニックネームつけて"]
selected_function = st.selectbox("機能を選択してください", function_options)

def generate_response(image):
    # 画像をbase64エンコード
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    try:
        if selected_function == "ボケる":
            user_prompt = "この写真についておかしな例えでボケてください。"
        elif selected_function == "褒める":

            user_prompt = "この写真についてハイテンションで褒めてください。"
        else: # ニックネームつける
            user_prompt = "この写真の人物にふさわしい少し変な面白おかしいニックネームをつけてください。動物や歴史上の人物なども可。二つ名、通り名、屋号も考えてください"

        # litellmを使用してボケ/褒めを生成
        response = completion(
 #           model=selected_model,  # 選択されたモデルを使用
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはアメリカの１０代の超軽いノリのギャルです。写真の画像を自分と勘違いしないようにしてください。必ず日本語で話します。決して相手を貶さないでください。いかなる場合でも相手を不快にさせることはしないでください。ボケる場合は、相手を不快にさせず、面白い例えを考えてください"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }
            ],
            max_tokens=200  # 応答の最大トークン数を設定
        )
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "AIからの返答がありませんでした。"
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return None

st.title("ボケて褒めてニックネームつけて")

# 画像ソースの選択
image_source = st.radio("画像ソースを選択してください", ["カメラ撮影", "ファイルアップロード"])

if image_source == "ファイルアップロード":
    uploaded_file = st.file_uploader("写真をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードした写真', use_column_width=True)
        response = generate_response(image)
        if response:
            st.write(response)

elif image_source == "カメラ撮影":
    camera_image = st.camera_input("写真を撮影してください")
    if camera_image is not None:
        image = Image.open(camera_image)
    #    image = image.resize((image.width // 3, image.height // 3))  # 画像のサイズを1/3に変更
    #    st.image(image, caption='撮影した写真', use_column_width=True)
        response = generate_response(image)
        if response:
            st.write(response)