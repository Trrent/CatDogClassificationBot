import asyncio
import logging
import os
import configparser

import numpy as np
import tensorflow as tf
import cv2


from aiogram import Bot, Dispatcher, types, F

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read('config.ini')

TOKEN = config['Telegram']['token']

bot = Bot(TOKEN)
dp = Dispatcher()

IMG_SIZE = 100
model = tf.keras.models.load_model('saved_model/saved_model')

CATEGORIES = ['Dog', 'Cat']


@dp.message(F.photo)
async def prediction(message: types.Message, bot: Bot):
    path = f"tmp/{message.photo[-1].file_id}.jpg"
    await bot.download(message.photo[-1], destination=path)

    img_array = cv2.imread(path)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    x = np.asarray([np.array(new_array)])
    pred = model.predict(x).round()
    await message.answer(f"I think it is {CATEGORIES[np.argmax(pred)]}")
    os.remove(path)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
