%% Comparación de tiempos de ejecución de modelos
% Nombres de los modelos
model_names = {'RealESRGAN', 'HAT', 'DRTC', 'HMA', 'WaveMix', 'StableSR'};

% Tiempos proporcionados (en segundos) y conversión a minutos
times_sec = [386.05, 179.86, 517.46, 244.76, 49.69, 916.95];
times_min = times_sec / 60;

% Gráfica de barras de los tiempos
figure;
bar(times_min, 'FaceColor', [0.2 0.6 0.5]);
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'FontSize', 12);
ylabel('Tiempo (minutos)', 'FontSize', 12);
title('Comparación de Tiempos de Ejecución de Modelos', 'FontSize', 14);
grid on;

%% Cálculo de Métricas a partir de los datasets

% 1) Directorios de los datasets procesados (uno por modelo)
datasets = {
    "C:\Users\nekos\OneDrive\Escritorio\RealESRGAN"
    "C:\Users\nekos\OneDrive\Escritorio\HAT"
    "C:\Users\nekos\OneDrive\Escritorio\DRTC"
    "C:\Users\nekos\OneDrive\Escritorio\HMA"
    "C:\Users\nekos\OneDrive\Escritorio\WaveMix"
    "C:\Users\nekos\OneDrive\Escritorio\StableSR"
};

num_models = length(datasets);

% 2) Inicializamos una estructura para almacenar los valores individuales de cada métrica
metricValues.PSNR     = cell(1, num_models);
metricValues.SSIM     = cell(1, num_models);
metricValues.MS_SSIM  = cell(1, num_models);
metricValues.FSIM     = cell(1, num_models);
metricValues.BRISQUE  = cell(1, num_models);
metricValues.PIQE     = cell(1, num_models);

% 3) Directorio de las imágenes de referencia
ref_dir = "C:\Users\nekos\OneDrive\Escritorio\Referencia";

% 4) Bucle para procesar cada dataset (modelo)
for model_idx = 1:num_models
    dataset_dir = datasets{model_idx};
    
    % Obtener la lista de ficheros (se omiten los directorios)
    file_list = dir(fullfile(dataset_dir, '*.*'));
    file_list = file_list(~[file_list.isdir]);
    
    % Inicializar acumuladores de métricas para el modelo actual
    psnr_accum    = [];
    ssim_accum    = [];
    ms_ssim_accum = [];
    fsim_accum    = [];
    brisque_accum = [];
    piqe_accum    = [];
    
    % Procesar cada imagen en el dataset
    for i = 1:length(file_list)
        processed_name = file_list(i).name;
        processed_path = fullfile(dataset_dir, processed_name);
        
        % Intentar leer la imagen procesada
        try
            processed_img = imread(processed_path);
        catch
            warning('No se pudo leer la imagen: %s', processed_path);
            continue;
        end
        
        % EXTRAER EL ÍNDICE (hasta 2 dígitos) DEL NOMBRE DEL ARCHIVO
        tokens = regexp(processed_name, '^(\d{1,2})', 'tokens');
        if isempty(tokens)
            warning('No se pudo extraer el índice de la imagen: %s', processed_name);
            % Se calculan solo métricas SIN REFERENCIA (BRISQUE y PIQE)
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
            continue; % No se pueden calcular las métricas con referencia
        end
        
        img_index = str2double(tokens{1}{1});
        
        % Buscar la imagen de referencia considerando cualquier extensión.
        % Se busca un archivo que comience con el número (por ejemplo, "1.*")
        ref_pattern = fullfile(ref_dir, sprintf('%d.*', img_index));
        ref_info = dir(ref_pattern);
        if isempty(ref_info)
            warning('No se encontró la imagen de referencia para %d (patrón: %s)', img_index, ref_pattern);
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
        else
            % Se utiliza el primer archivo encontrado
            ref_file = fullfile(ref_dir, ref_info(1).name);
        end
        
        % Intentar leer la imagen de referencia
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
            processed_gray = imresize(processed_gray, [size(original_gray, 1), size(original_gray, 2)]);
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
        
    end % Fin del bucle de imágenes
    
    % Guardar los valores calculados para cada métrica en la estructura
    metricValues.PSNR{model_idx}    = psnr_accum;
    metricValues.SSIM{model_idx}    = ssim_accum;
    metricValues.MS_SSIM{model_idx} = ms_ssim_accum;
    metricValues.FSIM{model_idx}    = fsim_accum;
    metricValues.BRISQUE{model_idx} = brisque_accum;
    metricValues.PIQE{model_idx}    = piqe_accum;
    
    fprintf('Modelo %s procesado. Imágenes evaluadas: %d\n', model_names{model_idx}, length(file_list));
end

%% Creación de gráficos de Boxplots para cada métrica
% Lista de métricas (nombre de cada campo en la estructura)
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'BRISQUE', 'PIQE'};

for m = 1:length(metricsList)
    metricName = metricsList{m};
    
    % Concatenar los datos de todos los modelos y crear la variable de grupo
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
    % Sobreponer la media de cada modelo (marcada con un cuadrado rojo)
    for model_idx = 1:num_models
        data = metricValues.(metricName){model_idx};
        if ~isempty(data)
            m_val = mean(data);
            plot(model_idx, m_val, 'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end
    hold off;
    title(sprintf('Boxplot de %s para todos los modelos', metricName), 'FontSize', 14);
    ylabel(metricName, 'FontSize', 12);
    xlabel('Modelos', 'FontSize', 12);
    grid on;
end
