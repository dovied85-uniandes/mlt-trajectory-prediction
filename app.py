from flask import Flask, request, send_file
from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.image import decode_jpeg
import numpy as np
from l5kit.visualization import draw_trajectory
from l5kit.geometry import transform_points
from PIL import Image
import io
import sys


# Matriz para convertir de coordenadas agente (metros) a pixeles
# 2 pixeles son 1m, eje 'y' crece hacia abajo, y agente se centra en (56, 112):
RASTER_FROM_AGENT = np.array([[2., 0., 56.], [0., -2., 112.], [0., 0., 1.]])

# Se carga el modelo ya entrenado:
model = load_model('modelo_proyecto')



app = Flask(__name__)
CORS(app)



@app.route("/api/process_image", methods=["POST"])
def process_image():
	# Se lee la imagen del HTTP request:
	img = request.files["file"].read()
	img = decode_jpeg(img)
	
	# Se convierte a numpy tipo int:
	img = img_to_array(img).astype(np.uint8)

	# Se le pasa la imagen (escalada a [0,1]) al modelo y este arroja la trajectoria predicha
	# A los modelos de tensorflow siempre toca pasarle los datos por batch, entonces se agrega una primera dimensión dummy:
	traj_meters = model((img/255.).reshape(-1, 224, 224, 3))
	# La salida se pasa de tensor a numpy, y como está en batch, se saca el 1er (y único) elemento:
	traj_meters = traj_meters.numpy()[0]

	# Se convierte la trayectoria predicha de metros a pixeles:
	traj_pixels = transform_points(traj_meters, RASTER_FROM_AGENT)
	# Se dibuja la trayectoria sobre la imagen, con el color deseado:
	draw_trajectory(img, traj_pixels, (0, 255, 255))

	# Se exporta la imagen
	img = Image.fromarray(img) #.save("img_out.jpg")
	im_out = io.BytesIO()
	img.save(im_out, "JPEG")
	im_out.seek(0)

	return send_file(im_out, mimetype="image/jpeg")