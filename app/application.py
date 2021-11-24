from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
class HeimdallApp(App):

    def build(self):
        # Main layout components
        self.image = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify', size_hint=(1, 0.1))
        self.verification = Label(text="Verification Uninitiated", size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)

        return layout


if __name__ == '__main__':
    HeimdallApp().run()