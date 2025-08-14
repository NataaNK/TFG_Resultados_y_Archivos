% Configuración inicial
folderPath = 'C:\Users\nekos\OneDrive\Escritorio\MasOrange\aura-sr\natalia\scaled_diff_entradas';  % Ruta a la carpeta con las imágenes SR
originalFolderPath = 'C:\Users\nekos\OneDrive\Escritorio\MasOrange\Real-ESRGAN\natalia\originals';  % Ruta a la carpeta con las imágenes originales
imageFiles = dir(fullfile(folderPath, '*_out.jpg'));  % Buscar imágenes SR que terminan en '_out.jpg'

% Estructuras para almacenar valores agrupados
resolutionGroups = containers.Map();
psnrGroups = containers.Map();
ssimGroups = containers.Map();

for k = 1:length(imageFiles)
    % Nombre completo de la imagen SR
    imagePath = fullfile(imageFiles(k).folder, imageFiles(k).name);
    
    % Extraer el nombre base y la resolución de partida de la imagen SR
    nameParts = split(imageFiles(k).name, '_');
    baseName = strcat(nameParts{1}, '_', nameParts{2});  % Combinar las dos primeras partes (e.g., 'Interestelar_2160x2880')
    resolutionStr = nameParts{end-1};  % Resolución de partida antes de "_out"
    
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
    if isKey(resolutionGroups, resolutionStr)
        psnrGroups(resolutionStr) = [psnrGroups(resolutionStr), psnrValue];
        ssimGroups(resolutionStr) = [ssimGroups(resolutionStr), ssimValue];
    else
        resolutionGroups(resolutionStr) = resolutionStr;
        psnrGroups(resolutionStr) = psnrValue;
        ssimGroups(resolutionStr) = ssimValue;
    end
end

% Preparar datos finales para graficar
resolutions = keys(resolutionGroups);
meanPnsrValues = [];
meanSsimValues = [];
resolutionLabels = resolutions;

for i = 1:length(resolutions)
    resKey = resolutions{i};
    meanPnsrValues(end+1) = mean(psnrGroups(resKey));
    meanSsimValues(end+1) = mean(ssimGroups(resKey));
end

% Graficar PSNR
figure;
subplot(2, 1, 1);
plot(1:length(resolutions), meanPnsrValues, '-o');
xlabel('Resolución de partida');
ylabel('Media de PSNR');
title('Resolución de partida vs. Media de PSNR');
grid on;
xticks(1:length(resolutions));
xticklabels(resolutionLabels);
set(gca, 'XTickLabelRotation', 90);

% Añadir el texto encima del título general
text(0.5, 1.2, 'IMÁGENES SUPERESCALADAS A 1080x1440', 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

% Graficar SSIM
subplot(2, 1, 2);
plot(1:length(resolutions), meanSsimValues, '-o');
xlabel('Resolución de partida');
ylabel('Media de SSIM');
title('Resolución de partida vs. Media de SSIM');
grid on;
xticks(1:length(resolutions));
xticklabels(resolutionLabels);
set(gca, 'XTickLabelRotation', 90);
