import streamlit as st
from PIL import Image
import litellm
from litellm import completion
import base64
from io import BytesIO
import os
from PIL import ImageOps  # 画像処理のためのライブラリをインポート
import requests  # requestsライブラリをインポート

st.title("ボケて褒めてニックネームつけて")
# シークレットからAPIキーを取得
openai_api_key = st.secrets["api"]["OPENAI_API_KEY"]
xi_api_key = st.secrets["api"]["XI_API_KEY"]
VOICE_ID = "jsCqWAovK2LkecY7zXl4"

if openai_api_key is None:
    raise ValueError("OpenAI APIキーが設定されていません。")

# APIキーをlitellmに設定
litellm.api_key = openai_api_key



def text_to_speech(text):
    # テキストを音声に変換する関数
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Accept": "application/json",
        "xi-api-key": xi_api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.1,
            "similarity_boost": 0.8
        }
    }
    
    response = requests.post(tts_url, headers=headers, json=data, stream=True)
    
    if response.ok:
        audio_data = BytesIO(response.content)
        if st.button("音声を再生"):  # ユーザーが音声再生を許可するボタン
            st.audio(audio_data, format="audio/mp3")  # 音声を再生
     #   st.markdown("音声を再生するには上のプレイヤーを使用してください。")
     #   print("音声出力が成功しました。")
    else:
        st.error("音声の生成中にエラーが発生しました。")
        print(response.text)

    # ブラウザ互換性のための JavaScript
    st.markdown(
        """
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            const audioElements = document.getElementsByTagName('audio');
            if (audioElements.length > 0) {
                const latestAudio = audioElements[audioElements.length - 1];
                latestAudio.oncanplaythrough = () => {
                    console.log('Audio can play through.');
                };
                latestAudio.onerror = (e) => {
                    console.error('Error loading audio:', e);
                };
            }
        });
        </script>
        """,
        unsafe_allow_html=True
    )

# モデル選択のためのセレクタを追加
#model_options = ["claude-3-haiku-20240307","gpt-4o-mini", "claude-3-5-sonnet-20240620", "gemini-pro-vision"]
#selected_model = st.selectbox("使用するモデルを選択してください", model_options)

function_options = ["ボケて", "褒めて", "ニックネームつけて"]
selected_function = st.radio("機能を選択してください", function_options)
#function_options = ["ボケて", "褒めて", "ニックネームつけて"]
#selected_function = st.selectbox("機能を選択してください", function_options)

def generate_response(image):
    # 画像をbase64エンコード
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    try:
        if selected_function == "ボケて":
            user_prompt = "この写真についておかしな例えでボケてください。"
        elif selected_function == "褒めて":

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
            #max_tokens=200  # 応答の最大トークン数を設定
            max_tokens=100  # 応答の最大トークン数を設定
        )
        if response and response.choices:
            ai_response = response.choices[0].message.content


            text_to_speech(ai_response)  
            return response.choices[0].message.content
        else:
            return "AIからの返答がありませんでした。"
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return None



# 画像ソースの選択
image_source = st.radio("画像ソースを選択してください", ["カメラ撮影", "ファイルアップロード"])


def compress_image(image):
    # 画像を圧縮する関数
    original_size = image.size[0] * image.size[1] * 3  # おおよそのファイルサイズ（RGBの場合）
    if original_size > 100 * 1024:  # 100KB以上の場合
        image = image.convert("RGB")  # RGBに変換
        image = ImageOps.exif_transpose(image)  # EXIF情報に基づいて画像を回転
        
        # サイズを調整して100KB程度に圧縮
        while True:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)  # JPEG形式で保存し、品質を調整
            compressed_size = buffered.tell()  # 圧縮後のサイズを取得
            if compressed_size <= 100 * 1024:  # 100KB以下になったら終了
                break
            image = image.resize((image.size[0] * 3 // 4, image.size[1] * 3 // 4))  # サイズを縮小

    return image, original_size, compressed_size 




if image_source == "ファイルアップロード":
    uploaded_file = st.file_uploader("写真をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image, original_size, compressed_size = compress_image(image)  # 画像を圧縮
        st.image(image, caption='アップロードした写真', use_column_width=True)
     #   st.write(f"元の画像サイズ: {original_size / 1024:.2f} KB")  # 元のサイズを表示
      #  st.write(f"圧縮後の画像サイズ: {compressed_size / 1024:.2f} KB")  # 圧縮後のサイズを表示
        response = generate_response(image)
        if response:
            st.write(response)

elif image_source == "カメラ撮影":
    camera_image = st.camera_input("写真を撮影してください")
    if camera_image is not None:
        image = Image.open(camera_image)
        image, original_size, compressed_size = compress_image(image)  # 画像を圧縮
      #  st.write(f"元の画像サイズ: {original_size / 1024:.2f} KB")  # 元のサイズを表示
     #   st.write(f"圧縮後の画像サイズ: {compressed_size / 1024:.2f} KB")  # 圧縮後のサイズを表示
        response = generate_response(image)
        if response:
            st.write(response)