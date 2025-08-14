%% Comparación de tiempos de ejecución de modelos
% Definición de nombres de modelos y tiempos (en minutos)
model_names = {'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRTC', 'HMA', 'StableSR', 'IPG'};

% Tiempos proporcionados (se han convertido a minutos)
% Ejemplo: "6:53" = 6 + 53/60 minutos, etc.
times_min = [6 + 53/60, 61 + 4/60, 14 + 34/60, 2 + 12/60, 17 + 16/60, 20 + 41/60, 54 + 56/60, 60 + 14/60];

% Gráfica de barras de los tiempos
figure;
bar(times_min, 'FaceColor',[0.2 0.6 0.5]);
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'FontSize',12);
ylabel('Tiempo (minutos)', 'FontSize',12);
title('Comparación de Tiempos de Ejecución de Modelos', 'FontSize',14);
grid on;

%% Cálculo de Métricas a partir de los datasets

% 1) Directorios de los datasets procesados (uno por modelo)
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

num_models = length(datasets);

% 2) Inicializamos una estructura para almacenar los valores individuales de cada métrica
metricValues.PSNR     = cell(1, num_models);
metricValues.SSIM     = cell(1, num_models);
metricValues.MS_SSIM  = cell(1, num_models);
metricValues.FSIM     = cell(1, num_models);
metricValues.BRISQUE  = cell(1, num_models);
metricValues.PIQE     = cell(1, num_models);

% 3) Bucle para procesar cada dataset (modelo)
for model_idx = 1:num_models
    dataset_dir = datasets{model_idx};
    
    % Obtener la lista de ficheros (se omiten directorios)
    file_list = dir(fullfile(dataset_dir, '*.*'));
    file_list = file_list(~[file_list.isdir]);
    
    % Inicializar acumuladores de métricas para el modelo actual
    psnr_accum    = [];
    ssim_accum    = [];
    ms_ssim_accum = [];
    fsim_accum    = [];
    brisque_accum = [];
    piqe_accum    = [];
    
    % Procesamos cada imagen en el dataset
    for i = 1:length(file_list)
        processed_name = file_list(i).name;
        processed_path = fullfile(dataset_dir, processed_name);
        
        % Intentamos leer la imagen procesada
        try
            processed_img = imread(processed_path);
        catch
            warning('No se pudo leer la imagen: %s', processed_path);
            continue;
        end
        
        % EXTRAER EL ÍNDICE (hasta 2 dígitos) DEL NOMBRE DE ARCHIVO
        tokens = regexp(processed_name, '^(\d{1,2})', 'tokens');
        if isempty(tokens)
            warning('No se pudo extraer el índice de la imagen: %s', processed_name);
            % Calculamos solo métricas SIN REFERENCIA (BRISQUE y PIQE)
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = [brisque_accum, brisque_score];
                piqe_accum    = [piqe_accum, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue; % No se puede calcular las métricas con referencia
        end
        
        img_index = str2double(tokens{1}{1});
        
        % DETERMINAR LA CARPETA DE ORIGINALES SEGÚN EL NOMBRE
        if contains(processed_name, 'resized', 'IgnoreCase', true)
            ref_dir = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals";
            ref_file = fullfile(ref_dir, sprintf('%d.jpg', img_index));
        else
            ref_dir = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals_deformaciones";
            ref_file = fullfile(ref_dir, sprintf('%d_downscaled.jpg', img_index));
        end
        
        % Comprobamos si la imagen de referencia existe y la leemos
        if ~isfile(ref_file)
            warning('No se encontró la imagen de referencia: %s', ref_file);
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = [brisque_accum, brisque_score];
                piqe_accum    = [piqe_accum, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        end
        
        try
            original_img = imread(ref_file);
        catch
            warning('No se pudo leer la imagen de referencia: %s', ref_file);
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                brisque_accum = [brisque_accum, brisque_score];
                piqe_accum    = [piqe_accum, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        end
        
        % Convertir a escala de grises (si es necesario)
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
        if any(size(processed_gray) ~= size(original_gray))
            processed_gray = imresize(processed_gray, [size(original_gray,1) size(original_gray,2)]);
        end
        
        %% Cálculo de métricas CON REFERENCIA
        
        % PSNR
        try
            psnr_val = psnr(processed_gray, original_gray);
            psnr_accum = [psnr_accum, psnr_val];
        catch
            warning('Error al calcular PSNR para %s.', processed_name);
        end
        
        % SSIM
        try
            ssim_val = ssim(processed_gray, original_gray);
            ssim_accum = [ssim_accum, ssim_val];
        catch
            warning('Error al calcular SSIM para %s.', processed_name);
        end
        
        % MS-SSIM (si la función multissim existe)
        if exist('multissim', 'file')
            try
                ms_ssim_val = multissim(processed_gray, original_gray);
                ms_ssim_accum = [ms_ssim_accum, ms_ssim_val];
            catch
                warning('Error al calcular MS-SSIM para %s.', processed_name);
            end
        end
        
        % FSIM
        try
            fsim_val = FSIM(processed_gray, original_gray);
            fsim_accum = [fsim_accum, fsim_val];
        catch
            warning('Error al calcular FSIM para %s.', processed_name);
        end
        
        %% Cálculo de métricas SIN REFERENCIA
        
        try
            brisque_score = brisque(processed_gray);
            brisque_accum = [brisque_accum, brisque_score];
        catch
            warning('Error al calcular BRISQUE para %s.', processed_name);
        end
        
        try
            piqe_score = piqe(processed_gray);
            piqe_accum = [piqe_accum, piqe_score];
        catch
            warning('Error al calcular PIQE para %s.', processed_name);
        end
        
    end  % Fin del bucle de imágenes
    
    % Guardamos los valores calculados para cada métrica en la estructura
    metricValues.PSNR{model_idx}    = psnr_accum;
    metricValues.SSIM{model_idx}    = ssim_accum;
    metricValues.MS_SSIM{model_idx} = ms_ssim_accum;
    metricValues.FSIM{model_idx}    = fsim_accum;
    metricValues.BRISQUE{model_idx} = brisque_accum;
    metricValues.PIQE{model_idx}    = piqe_accum;
    
    fprintf('Modelo %s procesado. Imágenes evaluadas: %d\n', model_names{model_idx}, length(file_list));
    
end

%% Creación de gráficos de Boxplots para cada métrica
% Lista de métricas (el nombre de cada campo en la estructura)
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'BRISQUE', 'PIQE'};

for m = 1:length(metricsList)
    metricName = metricsList{m};
    
    % Concatenamos los datos de todos los modelos y creamos la variable de grupo
    allData = [];
    groups  = [];
    for model_idx = 1:num_models
        data = metricValues.(metricName){model_idx};
        allData = [allData, data];
        groups  = [groups, repmat(model_idx, 1, length(data))];
    end
    
    figure;
    boxplot(allData, groups, 'Labels', model_names);
    hold on;
    % Sobreponer la media de cada modelo (se marca con un cuadrado rojo)
    for model_idx = 1:num_models
        data = metricValues.(metricName){model_idx};
        if ~isempty(data)
            m_val = mean(data);
            plot(model_idx, m_val, 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
        end
    end
    hold off;
    title(sprintf('Boxplot de %s para todos los modelos', metricName), 'FontSize',14);
    ylabel(metricName, 'FontSize',12);
    xlabel('Modelos', 'FontSize',12);
    grid on;
end
