#!/usr/bin/env python
# coding: utf-8

# In[ ]:
opencv-python-headless

import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_images(l, titles=None, fontsize=12):
    n = len(l)
    fig, ax = plt.subplots(1, n)
    for i, im in enumerate(l):
        if n == 1:
            ax.imshow(im)
            ax.axis('off')
            if titles is not None:
                ax.set_title(titles[i], fontsize=fontsize)
        else:
            ax[i].imshow(im)
            ax[i].axis('off')
            if titles is not None:
                ax[i].set_title(titles[i], fontsize=fontsize)
    fig.set_size_inches(fig.get_size_inches() * max(n, 2))
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Video Frame Analyzer")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        vid = cv2.VideoCapture(uploaded_file.name)

        frames = []
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            frames.append(frame)

        vid.release()
        st.write(f"Total frames: {len(frames)}")

        # Convertir a escala de grises y calcular diferencias
        bwframes = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in frames]
        diffs = [(p2 - p1) for p1, p2 in zip(bwframes[:-1], bwframes[1:])]
        diff_amps = np.array([np.linalg.norm(x) for x in diffs])

        # Gráfico de diferencias
        st.line_chart(diff_amps)

if __name__ == "__main__":
    main()

