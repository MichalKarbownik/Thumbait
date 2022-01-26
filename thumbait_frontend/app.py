import os
import pickle
from PIL import Image
import streamlit
import requests
import random
from logger import get_logger

logger = get_logger(__name__)
WITH_IMG = 640

footer = """<style>
a:link , a:visited{
color: gray;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<a style='display: block; text-align: center;' href="https://github.com/Raziel090" target="_blank">Michał Matusz</a>
<a style='display: block; text-align: center;' href="https://github.com/MichalKarbownik" target="_blank">Michał Karbownik</a>
<a style='display: block; text-align: center;' href="https://github.com/pszelew" target="_blank">Patryk Szelewski</a>
<br>
<p><a style='display: block; text-align: center;' href="https://github.com/MichalKarbownik/Thumbait" target="_blank">Project on GitHub</a></p>
</div>
"""


class SentimentApp:
    # def __init__(self, base_url="http://main-server:8080/predict"):
    def __init__(self, base_url="http://localhost:8080/predict"):
        self.base_url = base_url

    def __call__(self) -> None:
        streamlit.image("img/static/YouBait.png", width=WITH_IMG // 2)
        streamlit.title("Thumbait - is this project a bait?")
        # Using the "with" syntax
        with streamlit.form(key="my_form"):
            streamlit.text("Is this video a clickbait?")
            text_input = streamlit.text_input(label="Enter video_url or video_id")
            submit_button = streamlit.form_submit_button(label="Submit")
            if text_input:
                response = requests.get(self.base_url, params={"v": text_input}).json()
                link = response["link"]
                view_count = response["view_count"]
                view_count_pred = int(response["view_count_pred"])
                raw_output = response["raw_output"]
                output_trend = response["output_trend"]

                streamlit.video(link)

                streamlit.markdown(f"This video has **{view_count:,}** views")
                streamlit.markdown(
                    f"We think that it's thumbnail and title looks like video with **{view_count_pred:,}** views"
                )

                if view_count_pred >= view_count:
                    image_views = f"img/more/{random.choice(os.listdir(os.path.join('img/more')))}"
                    streamlit.markdown(
                        f"We think that it should have **{view_count_pred - view_count:,}** more views."
                    )
                    streamlit.markdown(
                        "Not a lot of views. It has a really nice thumbnail, though."
                    )
                else:
                    os.listdir(os.path.join("img/less"))
                    image_views = f"img/less/{random.choice(os.listdir(os.path.join('img/less')))}"
                    streamlit.markdown(
                        f"We think that it should have **{view_count - view_count_pred:,}** less views."
                    )
                    streamlit.markdown("Maybe thumbnail is not everything?")
                    streamlit.markdown(
                        "Content of a video matters too... or video reach."
                    )

                streamlit.image(image_views, width=WITH_IMG)

                streamlit.markdown(f"Interesing!")

                if output_trend == 1:
                    image_trend = f"img/bait/{random.choice(os.listdir(os.path.join('img/bait')))}"
                    streamlit.markdown(
                        f"We think that this video.... **Is a clickbait!**"
                    )
                    streamlit.markdown("It looks like a typical trending video")
                else:
                    image_trend = f"img/not_bait/{random.choice(os.listdir(os.path.join('img/not_bait')))}"
                    streamlit.markdown(
                        f"We think that this video.... **Is not a clickbait!**"
                    )
                    streamlit.markdown("It doesn't look like a typical trending video")

                streamlit.image(image_trend, width=WITH_IMG)

            streamlit.markdown(
                "**Authors**<br>"
                "[Michał Matusz](https://github.com/Raziel090)<br>"
                "[Michał Karbownik](https://github.com/MichalKarbownik)<br>"
                "[Patryk Szelewski](https://github.com/pszelew/)",
                unsafe_allow_html=True,
            )

            streamlit.markdown(
                "Source code<br>"
                "[Project on GitHub](https://github.com/MichalKarbownik/Thumbait)",
                unsafe_allow_html=True,
            )
            # streamlit.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    app = SentimentApp()
    app()
