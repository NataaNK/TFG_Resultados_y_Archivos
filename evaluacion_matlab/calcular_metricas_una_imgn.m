% Limpia el entorno
clear; clc; close all;

%% Selección del directorio de imágenes
folder = uigetdir(pwd, 'Seleccione el directorio con las imágenes a evaluar');
if folder == 0
    disp('No se seleccionó ningún directorio.');
    return;
end

%% Selección de la imagen de referencia
% Selección de la imagen de referencia con un filtro más amplio
[refName, refPath] = uigetfile({...
    '*.png;*.PNG;*.jpg;*.JPG;*.jpeg;*.JPEG;*.bmp;*.BMP','Imágenes soportadas (*.png, *.jpg, *.jpeg, *.bmp)'}, ...
    'Seleccione la imagen de referencia');
if isequal(refName, 0) || isequal(refPath, 0)
    error('No se seleccionó una imagen de referencia.');
end

% Construir la ruta completa del archivo
fullFileName = fullfile(refPath, refName);
disp(['Imagen de referencia seleccionada: ' fullFileName]);

% Verificar que el archivo existe
if ~exist(fullFileName, 'file')
    error('El archivo no existe o la ruta es incorrecta.');
end

% Intentar leer la imagen
try
    refImage = imread(fullFileName);
catch ME
    error('Error al leer la imagen de referencia: %s', ME.message);
end

%% Listado de imágenes en el directorio
% Se consideran varias extensiones
imgFiles = [dir(fullfile(folder, '*.jpg')); ...
            dir(fullfile(folder, '*.png')); ...
            dir(fullfile(folder, '*.jpeg')); ...
            dir(fullfile(folder, '*.bmp'))];

if isempty(imgFiles)
    disp('No se encontraron imágenes en el directorio.');
    return;
end

% Inicialización de variables para almacenar resultados
numImages = numel(imgFiles);
fileNames   = cell(numImages,1);
psnrValues  = zeros(numImages,1);
ssimValues  = zeros(numImages,1);
brisqueValues = zeros(numImages,1);
imagesCell  = cell(numImages,1);

%% Procesamiento de cada imagen
for k = 1:numImages
    fileName = imgFiles(k).name;
    filePath = fullfile(folder, fileName);
    img = imread(filePath);
    
    % Guardamos la imagen (para mostrarla después)
    imagesCell{k} = img;
    fileNames{k} = fileName;
    
    %% Asegurarse de que la imagen de prueba y la de referencia tengan el mismo tamaño para PSNR/SSIM
    if any(size(img,1:2) ~= size(refImage,1:2))
        imgForMetrics = imresize(img, [size(refImage,1), size(refImage,2)]);
    else
        imgForMetrics = img;
    end
    
    %% Cálculo de PSNR
    try
        psnrVal = psnr(imgForMetrics, refImage);
    catch
        psnrVal = NaN;
        warning('Error calculando PSNR para %s', fileName);
    end
    
    %% Cálculo de SSIM
    try
        ssimVal = ssim(imgForMetrics, refImage);
    catch
        ssimVal = NaN;
        warning('Error calculando SSIM para %s', fileName);
    end
    
    %% Cálculo de BRISQUE (se opera en escala de grises)
    if size(img,3) == 3
        imgGray = rgb2gray(img);
    else
        imgGray = img;
    end
    try
        brisqueVal = brisque(imgGray);
    catch
        brisqueVal = NaN;
        warning('Error calculando BRISQUE para %s', fileName);
    end
    
    % Guardar los valores calculados
    psnrValues(k)   = psnrVal;
    ssimValues(k)   = ssimVal;
    brisqueValues(k)= brisqueVal;
end

%% Creación de la ventana de resultados con pestañas (tabs)
fig = figure('Name','Resultados de Calidad de Imagen','NumberTitle','off','Units','normalized','Position',[0.1 0.1 0.8 0.8]);
tgroup = uitabgroup(fig);

%% Pestaña 1: Grid de imágenes con métricas
tab1 = uitab(tgroup, 'Title', 'Imágenes y Métricas');
% Calcula el número de filas y columnas para el grid
numCols = ceil(sqrt(numImages));
numRows = ceil(numImages/numCols);

% Usamos tiledlayout para organizar las imágenes en el grid
tiledlayout(tab1, numRows, numCols, 'TileSpacing','compact','Padding','compact');

for k = 1:numImages
    nexttile;
    imshow(imagesCell{k});
    % Se muestra el nombre de la imagen y los valores (PSNR, SSIM y BRISQUE)
    title({fileNames{k}, ...
           sprintf('PSNR: %.2f dB', psnrValues(k)), ...
           sprintf('SSIM: %.4f', ssimValues(k)), ...
           sprintf('BRISQUE: %.2f', brisqueValues(k))}, 'FontSize',8);
end

%% Pestaña 2: Tabla con los resultados
tab2 = uitab(tgroup, 'Title', 'Tabla de Métricas');
% Se arma la matriz de datos para la tabla
data = cell(numImages, 4);
for k = 1:numImages
    data{k,1} = fileNames{k};
    data{k,2} = psnrValues(k);
    data{k,3} = ssimValues(k);
    data{k,4} = brisqueValues(k);
end
columnNames = {'Imagen', 'PSNR (dB)', 'SSIM', 'BRISQUE'};

% Se crea la tabla ocupando todo el espacio de la pestaña
uitable('Parent', tab2, 'Data', data, 'ColumnName', columnNames, ...
    'Units', 'normalized', 'Position', [0 0 1 1]);
