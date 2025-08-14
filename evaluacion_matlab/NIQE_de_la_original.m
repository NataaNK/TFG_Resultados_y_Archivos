% Ruta de la imagen
imagePath = 'C:\Users\nekos\OneDrive\Escritorio\MasOrange\Real-ESRGAN\natalia\originals\Interestelar_2160x2880.jpg';  % Reemplaza con la ruta de tu imagen

% Leer la imagen
img = imread(imagePath);

% Calcular NIQE
niqeValue = niqe(img);

% Imprimir el valor de NIQE
fprintf('El valor de NIQE de la imagen es: %.2f\n', niqeValue);
