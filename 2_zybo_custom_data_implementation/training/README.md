# utils

Para entrenar la red final y convertir el modelo hls4ml a proyecto de Vivado

1. Correr: $ python3 -m pip install requirements.txt
2. Ajustar path de vivado "os.environ['PATH'] = '/mnt/shared/Vivado/2019.2/bin:' + os.environ['PATH']" en training.py
3. Ejecutar setup_zybo_hls4ml.sh
4. Correr: $ python3 training.py
