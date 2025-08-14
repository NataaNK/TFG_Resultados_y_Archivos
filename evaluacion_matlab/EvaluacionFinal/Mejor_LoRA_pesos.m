%% Script: Cálculo de mejoras entre condición X y múltiples condiciones Y (subcarpetas)
%
% Este script compara la condición X (fija) con cada subcarpeta encontrada en
% las carpetas de "LoRA". La comparación se realiza sobre las métricas:
%   - PSNR, SSIM, MS_SSIM, FSIM (donde valores mayores son mejores)
%   - BRISQUE, PIQE (donde valores menores son mejores)
%
% Se muestran al final:
%   1. Las subcarpetas que mejoran TODAS las métricas (ninguna mejora negativa).
%   2. Si ninguna mejora TODAS, aquellas que mejoran 5 métricas (solo una negativa).
%   3. La subcarpeta con mayor mejora global.
%
% NOTA: Las fórmulas de mejora se definen como:
%   Para métricas "mayor es mejor": improvement = ((mean_Y - mean_X)/mean_X)*100.
%   Para métricas "menor es mejor": improvement = ((mean_X - mean_Y)/mean_X)*100.

%% Parámetros y rutas
% Condición X (sin cambios)
path_X = 'C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\x4v3wdn_default';
ref_dir = 'C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\FinetuningDataset\ConTexto100Original';

% Raíz de carpetas para condición Y (LoRA)
lora_root = 'C:\Users\nekos\OneDrive\Escritorio\LoRA';

% Lista de métricas a comparar
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'BRISQUE', 'PIQE'};

%% 1. Procesamiento de la condición X
% Inicializar estructura para almacenar las métricas de X
metricValues_X.PSNR    = [];
metricValues_X.SSIM    = [];
metricValues_X.MS_SSIM = [];
metricValues_X.FSIM    = [];
metricValues_X.BRISQUE = [];
metricValues_X.PIQE    = [];
metricValues_X.DIST_BRISQUE = [];
metricValues_X.DIST_PIQE    = [];

files_X = dir(fullfile(path_X, '*.*'));
files_X = files_X(~[files_X.isdir]);

for i = 1:length(files_X)
    proc_name = files_X(i).name;
    proc_path = fullfile(path_X, proc_name);
    try
        proc_img = imread(proc_path);
    catch
        warning('No se pudo leer la imagen: %s', proc_path);
        continue;
    end
    
    % Extraer índice (hasta 2 dígitos) del nombre de archivo
    tokens = regexp(proc_name, '^(\d{1,2})', 'tokens');
    if isempty(tokens)
        warning('No se pudo extraer el índice de la imagen: %s', proc_name);
        proc_gray = toGray(proc_img);
        try
            % Solo cálculo sin referencia
            metricValues_X.BRISQUE(end+1) = brisque(proc_gray);
            metricValues_X.PIQE(end+1)    = piqe(proc_gray);
        catch
            warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
        end
        continue;
    end
    img_index = str2double(tokens{1}{1});
    
    % Determinar la palabra clave (BN, Color o Sepia)
    if contains(proc_name, 'BN')
        keyword = 'BN';
    elseif contains(proc_name, 'Color')
        keyword = 'Color';
    elseif contains(proc_name, 'Sepia')
        keyword = 'Sepia';
    else
        warning('No se encontró palabra clave en %s', proc_name);
        proc_gray = toGray(proc_img);
        try
            metricValues_X.BRISQUE(end+1) = brisque(proc_gray);
            metricValues_X.PIQE(end+1)    = piqe(proc_gray);
        catch
            warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
        end
        continue;
    end
    
    % Buscar imagen de referencia en ref_dir
    ref_files = dir(fullfile(ref_dir, '*.*'));
    pattern = sprintf('^%d_%s.*', img_index, keyword);
    matches = arrayfun(@(x) ~isempty(regexp(x.name, pattern, 'once')), ref_files);
    ref_info = ref_files(matches);
    if isempty(ref_info)
        warning('No se encontró imagen de referencia para %d con keyword %s.', img_index, keyword);
        proc_gray = toGray(proc_img);
        try
            metricValues_X.BRISQUE(end+1) = brisque(proc_gray);
            metricValues_X.PIQE(end+1)    = piqe(proc_gray);
        catch
            warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
        end
        continue;
    else
        ref_file = fullfile(ref_dir, ref_info(1).name);
    end
    
    % Lectura de la imagen de referencia
    try
        orig_img = imread(ref_file);
    catch
        warning('No se pudo leer la imagen de referencia: %s', ref_file);
        proc_gray = toGray(proc_img);
        try
            metricValues_X.BRISQUE(end+1) = brisque(proc_gray);
            metricValues_X.PIQE(end+1)    = piqe(proc_gray);
        catch
            warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
        end
        continue;
    end
    
    % Conversión a escala de grises y ajuste de tamaño
    proc_gray = toGray(proc_img);
    orig_gray = toGray(orig_img);
    if any(size(proc_gray) ~= size(orig_gray))
        proc_gray = imresize(proc_gray, [size(orig_gray,1), size(orig_gray,2)]);
    end
    
    % Cálculo de métricas CON REFERENCIA
    try
        metricValues_X.PSNR(end+1) = psnr(proc_gray, orig_gray);
    catch
        warning('Error al calcular PSNR para %s.', proc_name);
    end
    try
        metricValues_X.SSIM(end+1) = ssim(proc_gray, orig_gray);
    catch
        warning('Error al calcular SSIM para %s.', proc_name);
    end
    if exist('multissim','file')
        try
            metricValues_X.MS_SSIM(end+1) = multissim(proc_gray, orig_gray);
        catch
            warning('Error al calcular MS_SSIM para %s.', proc_name);
        end
    end
    try
        metricValues_X.FSIM(end+1) = FSIM(proc_gray, orig_gray);
    catch
        warning('Error al calcular FSIM para %s.', proc_name);
    end
    
    % Cálculo de métricas SIN REFERENCIA y distancias
    try
        orig_brisque = brisque(orig_gray);
    catch
        warning('Error al calcular BRISQUE en original para %s.', proc_name);
        orig_brisque = NaN;
    end
    try
        orig_piqe = piqe(orig_gray);
    catch
        warning('Error al calcular PIQE en original para %s.', proc_name);
        orig_piqe = NaN;
    end
    
    try
        proc_brisque = brisque(proc_gray);
        metricValues_X.BRISQUE(end+1) = proc_brisque;
        if ~isnan(orig_brisque)
            metricValues_X.DIST_BRISQUE(end+1) = abs(proc_brisque - orig_brisque);
        end
    catch
        warning('Error al calcular BRISQUE para %s.', proc_name);
    end
    try
        proc_piqe = piqe(proc_gray);
        metricValues_X.PIQE(end+1) = proc_piqe;
        if ~isnan(orig_piqe)
            metricValues_X.DIST_PIQE(end+1) = abs(proc_piqe - orig_piqe);
        end
    catch
        warning('Error al calcular PIQE para %s.', proc_name);
    end
end

% Inicializar estructuras para medias y desviaciones de X
mean_X = struct();
std_X  = struct();

% Cálculo de medias y desviaciones de X para cada métrica
for m = 1:length(metricsList)
    field = metricsList{m};
    mean_X.(field) = mean(metricValues_X.(field));
    std_X.(field)  = std(metricValues_X.(field));
end

%% 2. Recorrido de las subcarpetas de condición Y en LoRA
% Se obtiene la lista de carpetas (por ejemplo, L1, L1_GAN, etc.)
lora_folders = dir(lora_root);
lora_folders = lora_folders([lora_folders.isdir] & ~ismember({lora_folders.name},{'.','..'}));

% Se buscarán las subcarpetas dentro de cada carpeta de LoRA
candidatePaths = {};
for i = 1:length(lora_folders)
    currentFolder = fullfile(lora_root, lora_folders(i).name);
    subFolders = dir(currentFolder);
    subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name},{'.','..'}));
    for j = 1:length(subFolders)
        candidatePaths{end+1} = fullfile(currentFolder, subFolders(j).name);
    end
end

% Estructura para almacenar resultados por cada subcarpeta (candidato Y)
results = struct('folderPath',{},'improvements',{},'overall_improvement',{},'numNegatives',{});

%% 3. Procesamiento de cada subcarpeta Y
for k = 1:length(candidatePaths)
    yFolder = candidatePaths{k};
    fprintf('Procesando carpeta Y: %s\n', yFolder);
    
    % Inicializar estructura para métricas de Y
    metricValues_Y.PSNR    = [];
    metricValues_Y.SSIM    = [];
    metricValues_Y.MS_SSIM = [];
    metricValues_Y.FSIM    = [];
    metricValues_Y.BRISQUE = [];
    metricValues_Y.PIQE    = [];
    metricValues_Y.DIST_BRISQUE = [];
    metricValues_Y.DIST_PIQE    = [];
    
    files_Y = dir(fullfile(yFolder, '*.*'));
    files_Y = files_Y(~[files_Y.isdir]);
    
    % Procesar cada imagen en la subcarpeta Y
    for i = 1:length(files_Y)
        proc_name = files_Y(i).name;
        proc_path = fullfile(yFolder, proc_name);
        try
            proc_img = imread(proc_path);
        catch
            warning('No se pudo leer la imagen: %s', proc_path);
            continue;
        end
        
        tokens = regexp(proc_name, '^(\d{1,2})', 'tokens');
        if isempty(tokens)
            warning('No se pudo extraer índice en %s', proc_name);
            proc_gray = toGray(proc_img);
            try
                metricValues_Y.BRISQUE(end+1) = brisque(proc_gray);
                metricValues_Y.PIQE(end+1)    = piqe(proc_gray);
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
            end
            continue;
        end
        img_index = str2double(tokens{1}{1});
        
        % Determinar palabra clave
        if contains(proc_name, 'BN')
            keyword = 'BN';
        elseif contains(proc_name, 'Color')
            keyword = 'Color';
        elseif contains(proc_name, 'Sepia')
            keyword = 'Sepia';
        else
            warning('No se encontró palabra clave en %s', proc_name);
            keyword = '';
        end
        
        % Buscar imagen original de referencia (si se tiene keyword se incluye; sino, por índice)
        if ~isempty(keyword)
            ref_pattern = fullfile(ref_dir, sprintf('%d_%s.*', img_index, keyword));
        else
            ref_pattern = fullfile(ref_dir, sprintf('%d.*', img_index));
        end
        ref_info = dir(ref_pattern);
        if isempty(ref_info)
            warning('No se encontró imagen de referencia para %d en %s', img_index, ref_dir);
            proc_gray = toGray(proc_img);
            try
                metricValues_Y.BRISQUE(end+1) = brisque(proc_gray);
                metricValues_Y.PIQE(end+1)    = piqe(proc_gray);
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
            end
            continue;
        else
            ref_file = fullfile(ref_dir, ref_info(1).name);
        end
        
        % Lectura de la imagen original y preprocesamiento
        try
            orig_img = imread(ref_file);
        catch
            warning('No se pudo leer imagen de referencia: %s', ref_file);
            proc_gray = toGray(proc_img);
            try
                metricValues_Y.BRISQUE(end+1) = brisque(proc_gray);
                metricValues_Y.PIQE(end+1)    = piqe(proc_gray);
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', proc_name);
            end
            continue;
        end
        
        proc_gray = toGray(proc_img);
        orig_gray = toGray(orig_img);
        if any(size(proc_gray) ~= size(orig_gray))
            proc_gray = imresize(proc_gray, [size(orig_gray,1) size(orig_gray,2)]);
        end
        
        % Cálculo de métricas CON REFERENCIA
        try
            metricValues_Y.PSNR(end+1) = psnr(proc_gray, orig_gray);
        catch
            warning('Error al calcular PSNR para %s.', proc_name);
        end
        try
            metricValues_Y.SSIM(end+1) = ssim(proc_gray, orig_gray);
        catch
            warning('Error al calcular SSIM para %s.', proc_name);
        end
        if exist('multissim','file')
            try
                metricValues_Y.MS_SSIM(end+1) = multissim(proc_gray, orig_gray);
            catch
                warning('Error al calcular MS_SSIM para %s.', proc_name);
            end
        end
        try
            metricValues_Y.FSIM(end+1) = FSIM(proc_gray, orig_gray);
        catch
            warning('Error al calcular FSIM para %s.', proc_name);
        end
        
        % Métricas SIN REFERENCIA y distancias
        try
            orig_brisque = brisque(orig_gray);
        catch
            warning('Error al calcular BRISQUE en original para %s.', proc_name);
            orig_brisque = NaN;
        end
        try
            orig_piqe = piqe(orig_gray);
        catch
            warning('Error al calcular PIQE en original para %s.', proc_name);
            orig_piqe = NaN;
        end
        
        try
            proc_brisque = brisque(proc_gray);
            metricValues_Y.BRISQUE(end+1) = proc_brisque;
            if ~isnan(orig_brisque)
                metricValues_Y.DIST_BRISQUE(end+1) = abs(proc_brisque - orig_brisque);
            end
        catch
            warning('Error al calcular BRISQUE para %s.', proc_name);
        end
        try
            proc_piqe = piqe(proc_gray);
            metricValues_Y.PIQE(end+1) = proc_piqe;
            if ~isnan(orig_piqe)
                metricValues_Y.DIST_PIQE(end+1) = abs(proc_piqe - orig_piqe);
            end
        catch
            warning('Error al calcular PIQE para %s.', proc_name);
        end
    end % fin de procesamiento de imágenes en carpeta Y
    
    % Verificar que se hayan obtenido datos (evitar división por cero)
    validCandidate = true;
    for m = 1:length(metricsList)
        if isempty(metricValues_Y.(metricsList{m}))
            validCandidate = false;
            break;
        end
    end
    if ~validCandidate
        fprintf('No se pudieron calcular algunas métricas para %s. Se omite.\n', yFolder);
        continue;
    end
    
    % Inicializar estructuras para medias y desviaciones de Y (¡IMPORTANTE!)
    mean_Y = struct();
    std_Y  = struct();
    
    % Cálculo de medias y desviaciones para Y
    for m = 1:length(metricsList)
        field = metricsList{m};
        mean_Y.(field) = mean(metricValues_Y.(field));
        std_Y.(field)  = std(metricValues_Y.(field));
    end
    
    % Cálculo de mejora (por métrica y global) respecto a X
    overall_weighted_sum = 0;
    total_weight = 0;
    improvements = struct();
    for m = 1:length(metricsList)
        field = metricsList{m};
        if ismember(field, {'PSNR','SSIM','MS_SSIM','FSIM'})
            base_impr = ((mean_Y.(field) - mean_X.(field)) / mean_X.(field)) * 100;
        else  % Para BRISQUE y PIQE (menor es mejor)
            base_impr = ((mean_X.(field) - mean_Y.(field)) / mean_X.(field)) * 100;
        end
        % Peso inverso de la media de desviaciones
        weight = 1 / (((std_X.(field) + std_Y.(field)) / 2) + eps);
        overall_weighted_sum = overall_weighted_sum + base_impr * weight;
        total_weight = total_weight + weight;
        improvements.(field) = base_impr;
    end
    overall_impr = overall_weighted_sum / total_weight;
    
    % Contar cuántas métricas tienen mejora negativa
    improvs = cell2mat(struct2cell(improvements));
    numNeg = sum(improvs < 0);
    
    % Guardar resultados del candidato
    results(end+1).folderPath = yFolder;
    results(end).improvements = improvements;
    results(end).overall_improvement = overall_impr;
    results(end).numNegatives = numNeg;
    
    fprintf('Carpeta: %s --- Mejora global: %.2f%%, Métricas negativas: %d\n', ...
        yFolder, overall_impr, numNeg);
end

%% 4. Selección e impresión de resultados
% 1. Subcarpetas que mejoran TODAS las métricas (ningún porcentaje negativo)
idx_all = find([results.numNegatives] == 0);

if ~isempty(idx_all)
    fprintf('\nSubcarpetas que mejoran TODAS las métricas:\n');
    for i = idx_all
        fprintf(' - %s\n', results(i).folderPath);
    end
else
    % 2. Si ninguna mejora todas, se muestran las que tienen solo un valor negativo (mejoran 5 de 6)
    idx_5 = find([results.numNegatives] == 1);
    fprintf('\nNinguna subcarpeta mejora todas las métricas.\n');
    fprintf('Subcarpetas que mejoran 5 de 6 métricas:\n');
    for i = idx_5
        fprintf(' - %s\n', results(i).folderPath);
    end
end

% 3. Subcarpeta con mayor mejora global
[~, idx_max] = max([results.overall_improvement]);
if ~isempty(idx_max)
    fprintf('\nSubcarpeta con mayor mejora global:\n%s\nMejora global: %.2f%%\n', ...
        results(idx_max).folderPath, results(idx_max).overall_improvement);
else
    fprintf('\nNo se procesaron subcarpetas válidas en condición Y.\n');
end

%% Función local para convertir a escala de grises
function out = toGray(img)
    % Si la imagen tiene 3 canales (por ejemplo, RGB), la convierte a escala de grises;
    % en caso contrario, devuelve la imagen original.
    if ndims(img)==3 && size(img,3)==3
        out = rgb2gray(img);
    else
        out = img;
    end
end
