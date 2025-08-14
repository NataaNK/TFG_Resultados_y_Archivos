% ============================================================
% (con y sin referencia) 
% ============================================================

% 1) Directorios de los datasets procesados
datasets = {
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\RealESRGAN_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\SWINFIR_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\HAT_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\AuraSR_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\DRTC_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\HMA_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\StableSR_caratulas100"
    "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\IPG_caratulas100"
};

% 2) Nombres de los modelos
model_names = {
    'Real-ESRGAN'
    'SwinFIR'
    'HAT'
    'Aura-SR'
    'DRTC'
    'HMA'
    'StableSR'
    'IPG'
};

% 3) Métricas de referencia que se van a calcular
metrics_ref = {'PSNR', 'SSIM', 'MS-SSIM', 'FSIM'};  

% 4) Métricas sin referencia
metrics_noref = {'BRISQUE', 'PIQE'};

% 5) Construimos la lista total de métricas y preparamos la matriz de resultados
all_metrics = [metrics_ref, metrics_noref];  % Unión de métricas con y sin referencia
num_metrics = length(all_metrics);
num_models  = length(datasets);
results     = zeros(num_metrics, num_models);

% ============================================================
% 6) Procesar cada dataset
% ============================================================

for model_idx = 1:num_models
    % Carpeta con los resultados del modelo actual
    dataset_dir = datasets{model_idx};
    
    % Obtenemos la lista de ficheros en la carpeta (tanto .jpg, .png, etc. como sea necesario)
    file_list = dir(fullfile(dataset_dir, '*.*'));
    % Filtramos posibles subdirectorios
    file_list = file_list(~[file_list.isdir]);
    
    % Inicializar acumuladores de métricas (y contadores para llevar la cuenta
    % de cuántas imágenes han podido calcular cada métrica con referencia).
    psnr_accum    = 0; psnr_count    = 0;
    ssim_accum    = 0; ssim_count    = 0;
    ms_ssim_accum = 0; ms_ssim_count = 0;
    fsim_accum    = 0; fsim_count    = 0;
    
    brisque_accum = 0; brisque_count = 0;
    piqe_accum    = 0; piqe_count    = 0;
    
    % --------------------------------------------------------
    % Recorremos cada imagen procesada en este dataset
    % --------------------------------------------------------
    for i = 1:length(file_list)
        processed_name = file_list(i).name;
        processed_path = fullfile(dataset_dir, processed_name);
        
        % Leer la imagen procesada
        try
            processed_img = imread(processed_path);
        catch
            warning('No se pudo leer la imagen procesada: %s', processed_path);
            continue;
        end
        
        % ----------------------------------------------------
        % EXTRAER EL ÍNDICE (hasta 2 dígitos) DEL NOMBRE DE ARCHIVO
        % El número al principio del nombre identificará la imagen de referencia
        % ----------------------------------------------------
        tokens = regexp(processed_name, '^(\d{1,2})', 'tokens');
        if isempty(tokens)
            warning('No se pudo extraer el índice de imagen en: %s', processed_name);
            % Aunque no tengamos índice, podemos calcular BRISQUE y PIQE
            % si deseamos, pero no tendremos referencia para PSNR, SSIM, etc.
            
            % Convertir a gris y computar BRISQUE y PIQE (sin referencia)
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            
            % Métricas sin referencia
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = brisque_accum + brisque_score;
                piqe_accum    = piqe_accum + piqe_score;
                brisque_count = brisque_count + 1;
                piqe_count    = piqe_count + 1;
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            
            continue; % No se puede calcular referencia
        end
        
        % Si sí hay tokens, tomamos el índice
        img_index = str2double(tokens{1}{1});
        
        % ----------------------------------------------------
        % DETERMINAR QUÉ CARPETA DE ORIGINALES USAR
        % según si en el nombre de la imagen aparece "resized" o no
        % ----------------------------------------------------
        if contains(processed_name, 'resized', 'IgnoreCase', true)
            ref_dir = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals";
            ref_file = fullfile(ref_dir, sprintf('%d.jpg', img_index));
        else
            ref_dir = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals_deformaciones";
            ref_file = fullfile(ref_dir, sprintf('%d_downscaled.jpg', img_index));
        end
        
        % Leemos la imagen de referencia si existe
        if ~isfile(ref_file)
            warning('No se encontró la imagen de referencia "%d.jpg" en: %s', ...
                img_index, ref_dir);
            
            % Sólo calculamos las métricas sin referencia
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = brisque_accum + brisque_score;
                piqe_accum    = piqe_accum + piqe_score;
                brisque_count = brisque_count + 1;
                piqe_count    = piqe_count + 1;
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            
            % Continuamos con la siguiente imagen
            continue;
        end
        
        try
            original_img = imread(ref_file);
        catch
            warning('No se pudo leer la imagen de referencia: %s', ref_file);
            % Sí calculamos las métricas sin referencia:
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = brisque_accum + brisque_score;
                piqe_accum    = piqe_accum + piqe_score;
                brisque_count = brisque_count + 1;
                piqe_count    = piqe_count + 1;
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            
            continue;
        end
        
        % ----------------------------------------------------
        % Convertir a escala de grises si es necesario
        % ----------------------------------------------------
        if size(original_img, 3) == 3
            original_gray = rgb2gray(original_img);
        else
            original_gray = original_img;
        end
        
        if size(processed_img, 3) == 3
            processed_gray = rgb2gray(processed_img);
        else
            processed_gray = processed_img;
        end
        
        % Ajustar tamaño si difieren
        if size(processed_gray,1) ~= size(original_gray,1) || ...
           size(processed_gray,2) ~= size(original_gray,2)
            processed_gray = imresize(processed_gray, [size(original_gray, 1), size(original_gray, 2)]);
        end
        
        % ----------------------------------------------------
        % Cálculo de métricas CON referencia
        % ----------------------------------------------------
        % PSNR
        psnr_val = psnr(processed_gray, original_gray);
        psnr_accum = psnr_accum + psnr_val;
        psnr_count = psnr_count + 1;
        
        % SSIM
        ssim_val = ssim(processed_gray, original_gray);
        ssim_accum = ssim_accum + ssim_val;
        ssim_count = ssim_count + 1;
        
        % MS-SSIM (si está disponible la función multissim)
        if exist('multissim', 'file')
            ms_ssim_val = multissim(processed_gray, original_gray);
            ms_ssim_accum = ms_ssim_accum + ms_ssim_val;
            ms_ssim_count = ms_ssim_count + 1;
        else
            warning('La función multissim no está disponible en tu MATLAB. Se omitirá MS-SSIM.');
        end
        
        % FSIM (asumiendo que FSIM está disponible como función)
        fsim_val = FSIM(processed_gray, original_gray);
        fsim_accum = fsim_accum + fsim_val;
        fsim_count = fsim_count + 1;
        
        % ----------------------------------------------------
        % Cálculo de métricas SIN referencia (BRISQUE, PIQE)
        % ----------------------------------------------------
        try
            brisque_score = brisque(processed_gray);
            piqe_score    = piqe(processed_gray);
            brisque_accum = brisque_accum + brisque_score;
            piqe_accum    = piqe_accum + piqe_score;
            brisque_count = brisque_count + 1;
            piqe_count    = piqe_count + 1;
        catch
            warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
        end
        
    end % fin del bucle de imágenes en el dataset
    
    % ============================================================
    % 7) Calcular los promedios para cada métrica en este dataset
    % ============================================================
    % Evitamos la división por 0 con if-else:
    if psnr_count > 0
        avg_psnr = psnr_accum / psnr_count;
    else
        avg_psnr = 0;
    end
    
    if ssim_count > 0
        avg_ssim = ssim_accum / ssim_count;
    else
        avg_ssim = 0;
    end
    
    if ms_ssim_count > 0
        avg_msssim = ms_ssim_accum / ms_ssim_count;
    else
        avg_msssim = 0;
    end
    
    if fsim_count > 0
        avg_fsim = fsim_accum / fsim_count;
    else
        avg_fsim = 0;
    end
    
    if brisque_count > 0
        avg_brisque = brisque_accum / brisque_count;
    else
        avg_brisque = 0;
    end
    
    if piqe_count > 0
        avg_piqe = piqe_accum / piqe_count;
    else
        avg_piqe = 0;
    end
    
    % Almacenamos los resultados en la matriz
    results(1, model_idx) = avg_psnr;    % PSNR
    results(2, model_idx) = avg_ssim;    % SSIM
    results(3, model_idx) = avg_msssim;  % MS-SSIM
    results(4, model_idx) = avg_fsim;    % FSIM
    results(5, model_idx) = avg_brisque; % BRISQUE
    results(6, model_idx) = avg_piqe;    % PIQE
end

% ============================================================
% 8) Crear tabla para visualización
% ============================================================
T = array2table(results, ...
    'RowNames', all_metrics, ...
    'VariableNames', model_names);

% Mostrar tabla en la ventana de comandos
disp(T);

% ============================================================
% 9) (Opcional) Crear una figura con uitable para visualizar
% ============================================================
figure;
uitable('Data', T{:,:}, ...
    'ColumnName', model_names, ...
    'RowName', all_metrics, ...
    'Position', [20 20 900 250]);

% (Opcional) Guardar la tabla como imagen
saveas(gcf, 'ModelComparisonTable_AllMetrics.png');
