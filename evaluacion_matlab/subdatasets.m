% =========================================================================
% Script para comparar métricas de calidad de 3 modelos (DRTC, HAT, RealESRGAN)
% en subcarpetas de 'COLOR' y 'ELEMENTOS'.
%
% Cada subcarpeta (p.ej. "DRTC_BlancoNegroBlur") indica:
%   - Modelo: DRTC
%   - Tipo/sufijo: BlancoNegroBlur
%
% Para medir métricas con referencia, apuntamos a la carpeta de originales,
% que se llama, por ejemplo:
%   C:\...\originals\COLOR\BlancoNegro\BlancoNegroOriginal
% =========================================================================

clc; clear; close all;

%% 1) Definir rutas de los modelos
base_path = 'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset';

model_names = {'DRTC','HAT','RealESRGAN'};
model_dirs = { ...
    fullfile(base_path, 'DRTC'), ...
    fullfile(base_path, 'HAT'), ...
    fullfile(base_path, 'RealESRGAN') ...
};

%% 2) Categorías y carpeta de originales
top_level_folders = {'COLOR','ELEMENTOS'};
originals_base = 'C:\Users\nekos\OneDrive\Escritorio\originals';

%% 3) Métricas a calcular
metrics_ref   = {'PSNR','SSIM','MS-SSIM','FSIM'};  
metrics_noref = {'BRISQUE','PIQE'};                
all_metrics   = [metrics_ref, metrics_noref];
num_metrics   = length(all_metrics);

%% 4) Función para calcular métricas en una carpeta dada
%   Se le pasa la carpeta con las imágenes SR, el "sufijo" (ej. "BlancoNegroBlur"),
%   y la carpeta de la categoría (p.ej. "...\originals\COLOR") para ubicar
%   la carpeta de referencia adecuada.
computeMetrics = @(folder_path, suffix_name, category_folder) ...
    compute_folder_metrics(folder_path, suffix_name, category_folder, metrics_ref, metrics_noref);

%% 5) Recorremos cada categoría
for c = 1:numel(top_level_folders)
    
    current_category = top_level_folders{c};  % 'COLOR' o 'ELEMENTOS'
    fprintf('\n=====================================\n');
    fprintf('Procesando categoría: %s\n', current_category);
    fprintf('=====================================\n');
    
    % Ruta de la carpeta de originales para la categoría (p.ej. "...\originals\COLOR")
    category_folder = fullfile(originals_base, current_category);
    if ~exist(category_folder, 'dir')
        warning('No existe la carpeta de originales: %s', category_folder);
        continue;
    end
    
    % ---------------------------------------------------------------------
    % 5.1) Buscar todas las subcarpetas en la carpeta de cada modelo
    %      (p.ej. "DRTC_BlancoNegroNoise", "HAT_BlancoNegroNoise"...)
    % ---------------------------------------------------------------------
    subfolders_all = {};
    for m = 1:numel(model_dirs)
        this_model_cat_path = fullfile(model_dirs{m}, current_category);
        if ~exist(this_model_cat_path, 'dir')
            continue; 
        end
        
        listing = dir(this_model_cat_path);
        for k = 1:numel(listing)
            if listing(k).isdir && ~strcmp(listing(k).name, '.') && ~strcmp(listing(k).name, '..')
                subfolders_all{end+1} = listing(k).name; %#ok<AGROW>
            end
        end
    end
    
    % ---------------------------------------------------------------------
    % 5.2) Convertir los nombres "DRTC_Sufijo", "HAT_Sufijo", etc. a "Sufijo" común
    % ---------------------------------------------------------------------
    suffix_list = cell(size(subfolders_all));
    for iSubf = 1:numel(subfolders_all)
        name_i = subfolders_all{iSubf};
        
        if startsWith(name_i, 'DRTC_')
            suffix_list{iSubf} = erase(name_i, 'DRTC_');
        elseif startsWith(name_i, 'HAT_')
            suffix_list{iSubf} = erase(name_i, 'HAT_');
        elseif startsWith(name_i, 'RealESRGAN_')
            suffix_list{iSubf} = erase(name_i, 'RealESRGAN_');
        else
            suffix_list{iSubf} = name_i;  % si no tiene prefijo
        end
    end
    
    unique_subfolders = unique(suffix_list);  % quitamos duplicados
    
    % ---------------------------------------------------------------------
    % 5.3) Para cada sufijo común, reconstruimos la carpeta de cada modelo
    % ---------------------------------------------------------------------
    for sfb = 1:numel(unique_subfolders)
        subf_suffix = unique_subfolders{sfb};  % p.ej. "BlancoNegroBlur"
        
        results_matrix = zeros(num_metrics, numel(model_dirs));
        any_model_found = false;
        
        for m = 1:numel(model_dirs)
            model_str  = model_names{m}; % 'DRTC', 'HAT', 'RealESRGAN'
            model_path = model_dirs{m};
            full_subf  = [model_str, '_', subf_suffix];
            
            this_folder = fullfile(model_path, current_category, full_subf);
            if ~exist(this_folder, 'dir')
                % No existe => dejamos métricas en 0
                continue;
            end
            
            any_model_found = true;
            
            % Calculamos métricas, pasándole el sufijo y la carpeta de la categoría
            metrics_vector = computeMetrics(this_folder, subf_suffix, category_folder);
            results_matrix(:, m) = metrics_vector(:);
        end
        
        if ~any_model_found
            continue;  % no hay subcarpetas para este sufijo
        end
        
        % Creamos tabla de resultados
        T = array2table(results_matrix, ...
            'RowNames', all_metrics, ...
            'VariableNames', model_names);
        
        fprintf('\n>>> Resultados para sufijo: "%s" (categoría: %s)\n', ...
            subf_suffix, current_category);
        disp(T);
    end
end


% =========================================================================
% 6) compute_folder_metrics:
%    Busca la carpeta de originales (<SubcarpetaOriginal>), recorre los
%    ficheros SR, extrae el índice, y localiza la imagen de referencia.
% =========================================================================
function metric_values = compute_folder_metrics(folder_path, suffix_name, category_folder, metrics_ref, metrics_noref)

    % 1) Hallamos la carpeta de referencia real, p.ej.:
    %    ...\BlancoNegro\BlancoNegroOriginal
    %    ...\Cartoon\CartoonOriginal
    ref_folder = parseReferenceFolder(suffix_name, category_folder);
    
    if ~exist(ref_folder, 'dir')
        warning('No existe carpeta de referencia: %s', ref_folder);
        % Todo saldrá 0 en las métricas con referencia.
    end
    
    % 2) Listamos ficheros procesados
    file_list = dir(fullfile(folder_path, '*.*'));
    file_list = file_list(~[file_list.isdir]);
    
    % Acumuladores de métricas
    psnr_accum    = 0; n_psnr    = 0;
    ssim_accum    = 0; n_ssim    = 0;
    ms_ssim_accum = 0; n_msssim  = 0;
    fsim_accum    = 0; n_fsim    = 0;
    
    brisque_accum = 0; n_brisque = 0;
    piqe_accum    = 0; n_piqe    = 0;
    
    for i = 1:numel(file_list)
        proc_name = file_list(i).name;
        proc_path = fullfile(folder_path, proc_name);
        
        % Leer imagen procesada
        proc_img = tryReadImage(proc_path);
        if isempty(proc_img)
            % No se pudo leer => pasamos a la siguiente
            continue;
        end
        
        % 3) Extraer índice (dígitos iniciales en el nombre)
        tokens = regexp(proc_name, '^(\d+)', 'tokens');
        if isempty(tokens)
            % Si no extraemos índice => solo métricas sin ref
            [bval, pval] = calcNoRef(proc_img);
            if ~isnan(bval), brisque_accum = brisque_accum + bval; n_brisque = n_brisque+1; end
            if ~isnan(pval), piqe_accum    = piqe_accum    + pval; n_piqe    = n_piqe+1; end
            continue;
        end
        
        idx = str2double(tokens{1}{1});
        
        % 4) Localizar y leer la imagen de referencia
        [ref_path, ref_img] = findReadableRefImage(ref_folder, idx);
        if isempty(ref_img)
            % No se pudo encontrar o leer la imagen de referencia
            [bval, pval] = calcNoRef(proc_img);
            if ~isnan(bval), brisque_accum = brisque_accum + bval; n_brisque = n_brisque+1; end
            if ~isnan(pval), piqe_accum    = piqe_accum    + pval; n_piqe    = n_piqe+1; end
            continue;
        end
        
        % Ajustar tamaño si difiere
        if size(proc_img,1) ~= size(ref_img,1) || size(proc_img,2) ~= size(ref_img,2)
            proc_img = imresize(proc_img, [size(ref_img,1), size(ref_img,2)]);
        end
        
        % Convertir a gris si hace falta
        if size(proc_img,3) == 3
            proc_gray = rgb2gray(proc_img);
        else
            proc_gray = proc_img;
        end
        
        if size(ref_img,3) == 3
            ref_gray = rgb2gray(ref_img);
        else
            ref_gray = ref_img;
        end
        
        % --- Métricas con referencia ---
        psnr_val = psnr(proc_gray, ref_gray);
        psnr_accum = psnr_accum + psnr_val; 
        n_psnr = n_psnr + 1;
        
        ssim_val = ssim(proc_gray, ref_gray);
        ssim_accum = ssim_accum + ssim_val; 
        n_ssim = n_ssim + 1;
        
        % MS-SSIM (si tienes multissim)
        if exist('multissim','file')
            ms_val = multissim(proc_gray, ref_gray);
            ms_ssim_accum = ms_ssim_accum + ms_val; 
            n_msssim = n_msssim + 1;
        end
        
        % FSIM (si tienes FSIM.m)
        if exist('FSIM','file')
            fs_val = FSIM(proc_gray, ref_gray);
            fsim_accum = fsim_accum + fs_val; 
            n_fsim = n_fsim + 1;
        end
        
        % --- Métricas sin referencia ---
        [bval, pval] = calcNoRef(proc_img);
        if ~isnan(bval), brisque_accum = brisque_accum + bval; n_brisque = n_brisque+1; end
        if ~isnan(pval), piqe_accum    = piqe_accum    + pval; n_piqe    = n_piqe+1; end
    end
    
    % 5) Promediar
    avg_psnr    = safeDiv(psnr_accum,    n_psnr);
    avg_ssim    = safeDiv(ssim_accum,    n_ssim);
    avg_msssim  = safeDiv(ms_ssim_accum, n_msssim);
    avg_fsim    = safeDiv(fsim_accum,    n_fsim);
    avg_brisque = safeDiv(brisque_accum, n_brisque);
    avg_piqe    = safeDiv(piqe_accum,    n_piqe);
    
    metric_values = [avg_psnr; avg_ssim; avg_msssim; avg_fsim; avg_brisque; avg_piqe];
end


% =========================================================================
% 7) parseReferenceFolder:
%    Dada la subcarpeta (p.ej. "BlancoNegroBlur") y la carpeta de categoría,
%    construye la ruta "...\BlancoNegro\BlancoNegroOriginal" (o la que toque).
% =========================================================================
function ref_folder = parseReferenceFolder(suffix_name, category_folder)
    % suffix_name = "BlancoNegroBlur", "CartoonNoise", ...
    % category_folder = "C:\...\originals\COLOR" o "...\ELEMENTOS"
    
    if contains(lower(suffix_name), 'blanconegro')
        subName = 'BlancoNegro';
        refName = 'BlancoNegroOriginal';
    elseif contains(lower(suffix_name), 'cartoon')
        subName = 'Cartoon';
        refName = 'CartoonOriginal';
    elseif contains(lower(suffix_name), 'dibujo')
        subName = 'Dibujo';
        refName = 'DibujoOriginal';
    elseif contains(lower(suffix_name), 'retrato')
        subName = 'Retrato';
        refName = 'RetratoOriginal';
    elseif contains(lower(suffix_name), 'fondoscargados')
        subName = 'FondosCargados';
        refName = 'FondoscargadosOriginal';
    elseif contains(lower(suffix_name), 'minimalista')
        subName = 'Minimalista';
        refName = 'MinimalistaOriginal';
    elseif contains(lower(suffix_name), 'sepia')
        subName = 'Sepia';
        refName = 'SepiaOriginal';
    elseif contains(lower(suffix_name), 'color')
        subName = 'Color';
        refName = 'ColorOriginal';
    else
        % Si no reconocemos, marcamos un "Desconocido"
        subName = 'Desconocido';
        refName = 'DesconocidoOriginal';
    end
    
    % Construir la ruta final
    ref_folder = fullfile(category_folder, subName, refName);
end

% =========================================================================
% 8) findReadableRefImage:
%    Dada la carpeta, busca el fichero (p.ej. "22.jpg", "22.png", etc.),
%    e intenta leerlo. Si no se puede (corrupto, etc.), lo omite y prueba
%    con otra extensión.
% =========================================================================
function [final_path, ref_img] = findReadableRefImage(ref_folder, idx)

    exts = {'.png','.jpg','.bmp','.tif'};  % ordena como prefieras
    final_path = '';
    ref_img    = [];  % si no se puede leer, se queda vacío
    
    if ~exist(ref_folder, 'dir')
        return; % no hay carpeta => devolvemos vacío
    end
    
    for iE = 1:numel(exts)
        candidate = fullfile(ref_folder, sprintf('%d%s', idx, exts{iE}));
        if exist(candidate, 'file')
            % Intentamos leer
            try
                tmp_img = imread(candidate);
                % Si sale bien:
                final_path = candidate;
                ref_img    = tmp_img;
                return;
            catch
                % Error al leer => continuamos con la siguiente extensión
                fprintf('Aviso: No se pudo leer "%s". Intentando otra extensión.\n', candidate);
                continue;
            end
        end
    end
end

% =========================================================================
% 9) tryReadImage:
%    Intenta imread de la imagen procesada. Si falla, devuelve [].
% =========================================================================
function img = tryReadImage(img_path)
    img = [];
    if ~exist(img_path,'file')
        fprintf('Aviso: No existe el archivo: %s\n', img_path);
        return;
    end
    
    try
        img = imread(img_path);
    catch ME
        fprintf('Aviso: Error al leer "%s". (%s)\n', img_path, ME.message);
        img = [];
    end
end

% =========================================================================
% 10) calcNoRef:
%    Calcula BRISQUE y PIQE sin referencia.
% =========================================================================
function [bval, pval] = calcNoRef(img)
    try
        if size(img,3) == 3
            g = rgb2gray(img);
        else
            g = img;
        end
        bval = brisque(g);
        pval = piqe(g);
    catch
        bval = NaN;
        pval = NaN;
    end
end

% =========================================================================
% 11) safeDiv:
%    Divide evitando NaN o /0
% =========================================================================
function val = safeDiv(num, den)
    if den == 0
        val = 0;
    else
        val = num / den;
    end
end
