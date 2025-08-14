% Configuración inicial
folderPath = 'C:\Users\nekos\OneDrive\Escritorio\MasOrange\aura-sr\natalia\scaled_diff_salidas';  % Ruta a la carpeta con las imágenes SR
originalFolderPath = 'C:\Users\nekos\OneDrive\Escritorio\MasOrange\Real-ESRGAN\natalia\originals';  % Ruta a la carpeta con las imágenes originales
imageFiles = dir(fullfile(folderPath, '*.jpg'));  % Buscar imágenes SR que terminan en '.jpg'

% Estructuras para almacenar valores agrupados
scaleGroups = containers.Map();
psnrGroups = containers.Map();
ssimGroups = containers.Map();

for k = 1:length(imageFiles)
    % Nombre completo de la imagen SR
    imagePath = fullfile(imageFiles(k).folder, imageFiles(k).name);
    
    % Extraer el nombre base y el factor de escala de la imagen SR
    nameParts = split(imageFiles(k).name, '_');
    baseName = strcat(nameParts{1}, '_', nameParts{2});  % Combinar las dos primeras partes (e.g., 'Interestelar_2160x2880')
    scaleStr = nameParts{end};  % Factor de escala después del último "_"
    scaleStr = erase(scaleStr, '.jpg');  % Remover la extensión '.jpg'
    scaleFactor = str2double(erase(scaleStr, 'x'));  % Convertir a número (por ejemplo, de 'x1.5' a 1.5)
    
    % Construir la ruta a la imagen original correspondiente
    realImagePath = fullfile(originalFolderPath, [baseName, '.jpg']);
    if ~isfile(realImagePath)
        warning('Imagen original no encontrada para %s. Saltando...', baseName);
        continue;
    end
    
    % Leer la imagen SR y la imagen original
    img = imread(imagePath);
    realImg = imread(realImagePath);
    
    % Redimensionar la imagen original para que coincida con la generada
    realImgResized = imresize(realImg, [size(img, 1), size(img, 2)]);
    
    % Calcular PSNR y SSIM
    psnrValue = psnr(img, realImgResized);
    ssimValue = ssim(img, realImgResized);
    
    % Almacenar los valores en los grupos
    scaleKey = num2str(scaleFactor);  % Convertir el factor de escala a cadena como clave del Map
    if isKey(scaleGroups, scaleKey)
        psnrGroups(scaleKey) = [psnrGroups(scaleKey), psnrValue];
        ssimGroups(scaleKey) = [ssimGroups(scaleKey), ssimValue];
    else
        scaleGroups(scaleKey) = scaleFactor;
        psnrGroups(scaleKey) = psnrValue;
        ssimGroups(scaleKey) = ssimValue;
    end
end

% Preparar datos finales para graficar
scaleFactors = cell2mat(values(scaleGroups));
[scaleFactors, sortIdx] = sort(scaleFactors);  % Ordenar factores de escala
scaleLabels = keys(scaleGroups);
scaleLabels = scaleLabels(sortIdx);  % Reordenar etiquetas

meanPnsrValues = [];
meanSsimValues = [];

for i = 1:length(scaleLabels)
    scaleKey = scaleLabels{i};
    meanPnsrValues(end+1) = mean(psnrGroups(scaleKey));
    meanSsimValues(end+1) = mean(ssimGroups(scaleKey));
end

% Graficar PSNR
figure;
subplot(2, 1, 1);
plot(scaleFactors, meanPnsrValues, '-o');
xlabel('Factor de Escala');
ylabel('Media de PSNR');
title('Factor de Escala vs. Media de PSNR');
grid on;
xticks(scaleFactors);
xticklabels(scaleLabels);
set(gca, 'XTickLabelRotation', 90);

% Añadir el texto encima del título general
text(0.5, 1.2, 'SR DE 480x640 POR DIFERENTES FACTORES', 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

% Graficar SSIM
subplot(2, 1, 2);
plot(scaleFactors, meanSsimValues, '-o');
xlabel('Factor de Escala');
ylabel('Media de SSIM');
title('Factor de Escala vs. Media de SSIM');
grid on;
xticks(scaleFactors);
xticklabels(scaleLabels);
set(gca, 'XTickLabelRotation', 90);
