%% Comparación de tiempos de ejecución de modelos
model_names = {'RealESRGAN', 'HAT', 'DRTC', 'HMA', 'WaveMix', 'StableSR'};
times_sec = [2124.40, 981.02, 2615.65, 1194.12, 166.98, 4014.30];
times_min = times_sec / 60;
figure;
bar(times_min, 'FaceColor', [0.2 0.6 0.5]);
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'FontSize', 12);
ylabel('Tiempo (minutos)', 'FontSize', 12);
title('Comparación de Tiempos de Ejecución de Modelos', 'FontSize', 14);
grid on;

%% Configuración de modelos y rutas
model_names = {'RealESRGAN', 'HAT', 'DRTC', 'HMA', 'WaveMix', 'StableSR'};

subdataset_template = {
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_BlancoNegroCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_ColorCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_SepiaCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\ELEMENTOS\DRTC_DibujoCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\ELEMENTOS\DRTC_ElemCartoonCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\ELEMENTOS\DRTC_FondosCargadosCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\ELEMENTOS\DRTC_MinimalistaCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\ELEMENTOS\DRTC_RetratoCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TEXTO\DRTC_ConTextoCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TEXTO\DRTC_SinTextoCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_CartoonCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_FuturistasCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_ManuscritasCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_MaquinaEscribirCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_PsicodelicasCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_SansSerifGruesasCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\TIPOGRAFIA\DRTC_SerifCompressed'
};

ref_dirs = {
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\BlancoNegro\BlancoNegroOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\Color\ColorOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\Sepia\SepiaOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\ELEMENTOS\Dibujo\DibujoOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\ELEMENTOS\Cartoon\CartoonOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\ELEMENTOS\FondosCargados\FondosCargadosOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\ELEMENTOS\Minimalista\MinimalistaOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\ELEMENTOS\Retrato\RetratoOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TEXTO\ConTexto\ConTextoOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TEXTO\SinTexto\SinTextoOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\Cartoon\CartoonOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\Futuristas\FuturistaOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\Manuscritas\ManuscritasOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\MaquinaEscribir\MaquinaEscribirOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\Psicodelicas\PsicodelicaOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\SansSerifGruesas\SansSerifGruesasOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\TIPOGRAFIA\Serif\SerifOriginal'
};


%% Inicializamos la estructura para almacenar los valores individuales de cada métrica (por modelo)
num_models = numel(model_names);
metricValues.PSNR    = cell(1, num_models);
metricValues.SSIM    = cell(1, num_models);
metricValues.MS_SSIM = cell(1, num_models);
metricValues.FSIM    = cell(1, num_models);
metricValues.BRISQUE = cell(1, num_models);
metricValues.PIQE    = cell(1, num_models);

%% Bucle para procesar cada modelo (acumulando resultados de todos sus subdatasets)
for model_idx = 1:num_models
    current_model = model_names{model_idx};
    fprintf('Procesando modelo: %s\n', current_model);
    
    % Genera las rutas de los subdatasets para este modelo reemplazando "DRTC" por el nombre del modelo
    dataset_paths = strrep(subdataset_template, "DRTC", current_model);
    
    % Inicializamos acumuladores para las métricas de este modelo
    psnr_accum    = [];
    ssim_accum    = [];
    ms_ssim_accum = [];
    fsim_accum    = [];
    brisque_accum = [];
    piqe_accum    = [];
    
    % Recorrer cada subdataset (y su carpeta de referencia correspondiente)
    for sub_idx = 1:length(dataset_paths)
        current_dataset = dataset_paths{sub_idx};
        current_ref_dir = ref_dirs{sub_idx};
        
        % Obtener la lista de imágenes en el subdataset (omitiendo directorios)
        file_list = dir(fullfile(current_dataset, '*.*'));
        file_list = file_list(~[file_list.isdir]);
        
        % Procesar cada imagen del subdataset
        for i = 1:length(file_list)
            processed_name = file_list(i).name;
            processed_path = fullfile(current_dataset, processed_name);
            
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
                % Calcular solo métricas SIN REFERENCIA (BRISQUE y PIQE)
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
            
            % Buscar la imagen de referencia en current_ref_dir (cualquier extensión)
            ref_pattern = fullfile(current_ref_dir, sprintf('%d.*', img_index));
            ref_info = dir(ref_pattern);
            if isempty(ref_info)
                warning('No se encontró la imagen de referencia para %d en %s', img_index, current_ref_dir);
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
                % Se utiliza el primer archivo encontrado en current_ref_dir
                ref_file = fullfile(current_ref_dir, ref_info(1).name);
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
            try
                psnr_val = psnr(processed_gray, original_gray);
                psnr_accum = [psnr_accum, psnr_val];
            catch
                warning('Error al calcular PSNR para %s.', processed_name);
            end
            
            try
                ssim_val = ssim(processed_gray, original_gray);
                ssim_accum = [ssim_accum, ssim_val];
            catch
                warning('Error al calcular SSIM para %s.', processed_name);
            end
            
            if exist('multissim', 'file')
                try
                    ms_ssim_val = multissim(processed_gray, original_gray);
                    ms_ssim_accum = [ms_ssim_accum, ms_ssim_val];
                catch
                    warning('Error al calcular MS-SSIM para %s.', processed_name);
                end
            end
            
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
        end % Fin del bucle de imágenes en el subdataset
    end % Fin del bucle de subdatasets para el modelo
    
    % Guardar los valores acumulados para cada métrica del modelo actual
    metricValues.PSNR{model_idx}    = psnr_accum;
    metricValues.SSIM{model_idx}    = ssim_accum;
    metricValues.MS_SSIM{model_idx} = ms_ssim_accum;
    metricValues.FSIM{model_idx}    = fsim_accum;
    metricValues.BRISQUE{model_idx} = brisque_accum;
    metricValues.PIQE{model_idx}    = piqe_accum;
    
    fprintf('Modelo %s procesado. Total imágenes evaluadas en todos los subdatasets: %d\n', current_model, numel(psnr_accum));
end

%% Creación de gráficos de Boxplots para cada métrica (acumulando todos los modelos)
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
