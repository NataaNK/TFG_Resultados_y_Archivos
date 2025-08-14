%% Comparación de resultados: CON NAFNet vs SIN NAFNet

% --- Definición de modelos ---
model_names = {'RealESRGAN', 'HAT', 'DRTC', 'HMA', 'WaveMix', 'StableSR'};
num_models = numel(model_names);

%% Gráfico comparativo de tiempos de ejecución
% Tiempos en segundos para cada experimento
times_sec_CON = [1138.27+519.86, 1138.27+177.60, 1138.27+708.01, 1138.27+226.99, 1138.27+28.42, 1138.27+687.06];
times_sec_SIN = [550.61, 214.87, 733.62, 300.50, 23.81, 728.36];

% Convertir a minutos
times_min_CON = times_sec_CON / 60;
times_min_SIN = times_sec_SIN / 60;

% Combinar en una matriz donde cada fila corresponde a un modelo y las columnas a CON y SIN
combined_times = [times_min_CON; times_min_SIN]'; 

% Gráfico de barras agrupadas
figure;
bar(combined_times);
set(gca, 'XTick', 1:num_models, 'XTickLabel', model_names, 'FontSize', 12);
ylabel('Tiempo (minutos)', 'FontSize', 12);
title('Comparación de Tiempos de Ejecución de Modelos', 'FontSize', 14);
legend('CON NAFNet', 'SIN NAFNet');
grid on;

%% Procesamiento de métricas para CON NAFNet
% Rutas para CON NAFNet
subdataset_template_CON = {...
    'C:\Users\nekos\OneDrive\Escritorio\NAFNet\DRTC_Final150\Final150Compressed'};
ref_dirs_CON = {...
    'C:\Users\nekos\OneDrive\Escritorio\NAFNet\Final150\Final150Original'};

% Inicializar estructura para almacenar las métricas
metricValues_CON.PSNR    = cell(1, num_models);
metricValues_CON.SSIM    = cell(1, num_models);
metricValues_CON.MS_SSIM = cell(1, num_models);
metricValues_CON.FSIM    = cell(1, num_models);
metricValues_CON.BRISQUE = cell(1, num_models);
metricValues_CON.PIQE    = cell(1, num_models);
% Se agregan los nuevos campos para las distancias (euclídeas)
metricValues_CON.DIST_BRISQUE = cell(1, num_models);
metricValues_CON.DIST_PIQE    = cell(1, num_models);

for model_idx = 1:num_models
    current_model = model_names{model_idx};
    fprintf('Procesando modelo (CON): %s\n', current_model);
    
    % Generar las rutas cambiando "DRTC" por el nombre del modelo
    dataset_paths = strrep(subdataset_template_CON, "DRTC", current_model);
    
    % Inicializar acumuladores para las métricas de este modelo
    psnr_accum    = [];
    ssim_accum    = [];
    ms_ssim_accum = [];
    fsim_accum    = [];
    brisque_accum = [];
    piqe_accum    = [];
    % Acumuladores para las distancias euclídeas en métricas sin referencia
    dist_BRISQUE_accum = [];
    dist_PIQE_accum    = [];
    
    % Procesar cada subdataset (en este ejemplo, solo hay uno)
    for sub_idx = 1:length(dataset_paths)
        current_dataset = dataset_paths{sub_idx};
        current_ref_dir = ref_dirs_CON{sub_idx};
        
        % Obtener la lista de imágenes (omitiendo carpetas)
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
                continue;
            end
            
            img_index = str2double(tokens{1}{1});
            
            % --- NUEVA LÓGICA PARA BUSCAR LA IMAGEN DE REFERENCIA ---
            % Extraer la palabra clave (BN, Color o Sepia) de la imagen procesada.
            if contains(processed_name, 'BN')
                keyword = 'BN';
            elseif contains(processed_name, 'Color')
                keyword = 'Color';
            elseif contains(processed_name, 'Sepia')
                keyword = 'Sepia';
            else
                warning('No se encontró palabra clave (BN, Color o Sepia) en %s', processed_name);
                % Calcular solo métricas SIN REFERENCIA:
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
            
            % Buscar la imagen de referencia en current_ref_dir.
            % Las imágenes de referencia tienen el formato: n_BN, n_Color o n_Sepia.
            ref_files = dir(fullfile(current_ref_dir, '*.*'));
            pattern = sprintf('^%d_%s.*', img_index, keyword); % Ejemplo: '1_BN.*'
            matches = arrayfun(@(x) ~isempty(regexp(x.name, pattern, 'once')), ref_files);
            ref_info = ref_files(matches);
            if isempty(ref_info)
                warning('No se encontró la imagen de referencia para %d con keyword %s en %s', img_index, keyword, current_ref_dir);
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
            % --- FIN NUEVA LÓGICA ---
            
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
            
            %% Cálculo de métricas SIN REFERENCIA Y DISTANCIA EUCLÍDEA
            % Primero, calcular la métrica sin referencia en la imagen original:
            try
                brisque_original = brisque(original_gray);
            catch
                warning('Error al calcular BRISQUE en imagen original para %s.', processed_name);
                brisque_original = NaN;
            end
            try
                piqe_original = piqe(original_gray);
            catch
                warning('Error al calcular PIQE en imagen original para %s.', processed_name);
                piqe_original = NaN;
            end
            
            % Calcular la métrica sin referencia en la imagen procesada y la distancia
            try
                brisque_score = brisque(processed_gray);
                brisque_accum = [brisque_accum, brisque_score];
                if ~isnan(brisque_original)
                    dist_br = abs(brisque_score - brisque_original);
                    dist_BRISQUE_accum = [dist_BRISQUE_accum, dist_br];
                end
            catch
                warning('Error al calcular BRISQUE para %s.', processed_name);
            end
            
            try
                piqe_score = piqe(processed_gray);
                piqe_accum = [piqe_accum, piqe_score];
                if ~isnan(piqe_original)
                    dist_piqe = abs(piqe_score - piqe_original);
                    dist_PIQE_accum = [dist_PIQE_accum, dist_piqe];
                end
            catch
                warning('Error al calcular PIQE para %s.', processed_name);
            end
        end % Fin del bucle de imágenes
    end % Fin del bucle de subdatasets para el modelo
    
    % Guardar los valores acumulados para cada métrica del modelo actual
    metricValues_CON.PSNR{model_idx}    = psnr_accum;
    metricValues_CON.SSIM{model_idx}    = ssim_accum;
    metricValues_CON.MS_SSIM{model_idx} = ms_ssim_accum;
    metricValues_CON.FSIM{model_idx}    = fsim_accum;
    metricValues_CON.BRISQUE{model_idx} = brisque_accum;
    metricValues_CON.PIQE{model_idx}    = piqe_accum;
    metricValues_CON.DIST_BRISQUE{model_idx} = dist_BRISQUE_accum;
    metricValues_CON.DIST_PIQE{model_idx}    = dist_PIQE_accum;
    
    fprintf('Modelo %s (CON) procesado. Total imágenes evaluadas: %d\n', current_model, numel(psnr_accum));
end

%% Procesamiento de métricas para SIN NAFNet
% Rutas para SIN NAFNet
subdataset_template_SIN = {...
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_BlancoNegroCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_ColorCompressed'
    'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset\DRTC\COLOR\DRTC_SepiaCompressed'};
ref_dirs_SIN = {...
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\BlancoNegro\BlancoNegroOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\Color\ColorOriginal'
    'C:\Users\nekos\OneDrive\Escritorio\IMAGENES\IMAGENES\Sub-datasets\COLOR\Sepia\SepiaOriginal'};

% Inicializar estructura para SIN
metricValues_SIN.PSNR    = cell(1, num_models);
metricValues_SIN.SSIM    = cell(1, num_models);
metricValues_SIN.MS_SSIM = cell(1, num_models);
metricValues_SIN.FSIM    = cell(1, num_models);
metricValues_SIN.BRISQUE = cell(1, num_models);
metricValues_SIN.PIQE    = cell(1, num_models);
% Nuevos campos para distancias
metricValues_SIN.DIST_BRISQUE = cell(1, num_models);
metricValues_SIN.DIST_PIQE    = cell(1, num_models);

for model_idx = 1:num_models
    current_model = model_names{model_idx};
    fprintf('Procesando modelo (SIN): %s\n', current_model);
    
    dataset_paths = strrep(subdataset_template_SIN, "DRTC", current_model);
    
    psnr_accum    = [];
    ssim_accum    = [];
    ms_ssim_accum = [];
    fsim_accum    = [];
    brisque_accum = [];
    piqe_accum    = [];
    dist_BRISQUE_accum = [];
    dist_PIQE_accum    = [];
    
    for sub_idx = 1:length(dataset_paths)
        current_dataset = dataset_paths{sub_idx};
        current_ref_dir = ref_dirs_SIN{sub_idx};
        file_list = dir(fullfile(current_dataset, '*.*'));
        file_list = file_list(~[file_list.isdir]);
        
        for i = 1:length(file_list)
            processed_name = file_list(i).name;
            processed_path = fullfile(current_dataset, processed_name);
            
            try
                processed_img = imread(processed_path);
            catch
                warning('No se pudo leer la imagen: %s', processed_path);
                continue;
            end
            
            tokens = regexp(processed_name, '^(\d{1,2})', 'tokens');
            if isempty(tokens)
                warning('No se pudo extraer el índice de la imagen: %s', processed_name);
                if size(processed_img,3)==3
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
            img_index = str2double(tokens{1}{1});
            
            % Buscar imagen de referencia (cualquier extensión)
            ref_pattern = fullfile(current_ref_dir, sprintf('%d.*', img_index));
            ref_info = dir(ref_pattern);
            if isempty(ref_info)
                warning('No se encontró la imagen de referencia para %d en %s', img_index, current_ref_dir);
                if size(processed_img,3)==3
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
                ref_file = fullfile(current_ref_dir, ref_info(1).name);
            end
            
            try
                original_img = imread(ref_file);
            catch
                warning('No se pudo leer la imagen de referencia: %s', ref_file);
                if size(processed_img,3)==3
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
            
            if size(original_img,3)==3
                original_gray = rgb2gray(original_img);
            else
                original_gray = original_img;
            end
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            
            if any(size(processed_gray) ~= size(original_gray))
                processed_gray = imresize(processed_gray, [size(original_gray,1) size(original_gray,2)]);
            end
            
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
            if exist('multissim','file')
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
            
            %% Cálculo de métricas SIN REFERENCIA Y DISTANCIA EUCLÍDEA
            try
                brisque_original = brisque(original_gray);
            catch
                warning('Error al calcular BRISQUE en imagen original para %s.', processed_name);
                brisque_original = NaN;
            end
            try
                piqe_original = piqe(original_gray);
            catch
                warning('Error al calcular PIQE en imagen original para %s.', processed_name);
                piqe_original = NaN;
            end
            
            try
                brisque_score = brisque(processed_gray);
                brisque_accum = [brisque_accum, brisque_score];
                if ~isnan(brisque_original)
                    dist_br = abs(brisque_score - brisque_original);
                    dist_BRISQUE_accum = [dist_BRISQUE_accum, dist_br];
                end
            catch
                warning('Error al calcular BRISQUE para %s.', processed_name);
            end
            
            try
                piqe_score = piqe(processed_gray);
                piqe_accum = [piqe_accum, piqe_score];
                if ~isnan(piqe_original)
                    dist_piqe = abs(piqe_score - piqe_original);
                    dist_PIQE_accum = [dist_PIQE_accum, dist_piqe];
                end
            catch
                warning('Error al calcular PIQE para %s.', processed_name);
            end
            
        end % fin de imágenes
    end % fin de subdatasets
    
    metricValues_SIN.PSNR{model_idx}    = psnr_accum;
    metricValues_SIN.SSIM{model_idx}    = ssim_accum;
    metricValues_SIN.MS_SSIM{model_idx} = ms_ssim_accum;
    metricValues_SIN.FSIM{model_idx}    = fsim_accum;
    metricValues_SIN.BRISQUE{model_idx} = brisque_accum;
    metricValues_SIN.PIQE{model_idx}    = piqe_accum;
    metricValues_SIN.DIST_BRISQUE{model_idx} = dist_BRISQUE_accum;
    metricValues_SIN.DIST_PIQE{model_idx}    = dist_PIQE_accum;
    
    fprintf('Modelo %s (SIN) procesado. Total imágenes evaluadas: %d\n', current_model, numel(psnr_accum));
end

%% Gráficos combinados de métricas (CON vs SIN)
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'BRISQUE', 'PIQE'};
for m = 1:length(metricsList)
    metricName = metricsList{m};
    figure; hold on;
    
    % Para cada modelo se grafican dos boxcharts: uno para CON (azul) y otro para SIN (rojo)
    for model_idx = 1:num_models
        pos_CON = model_idx - 0.15;  % Desplazamiento para separar visualmente
        pos_SIN = model_idx + 0.15;
        
        data_CON = metricValues_CON.(metricName){model_idx};
        data_SIN = metricValues_SIN.(metricName){model_idx};
        
        if ~isempty(data_CON)
            boxchart(repmat(pos_CON, size(data_CON)), data_CON, ...
                'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'b');
            % Graficar la media (cuadrado azul)
            plot(pos_CON, mean(data_CON), 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
        end
        if ~isempty(data_SIN)
            boxchart(repmat(pos_SIN, size(data_SIN)), data_SIN, ...
                'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'r');
            % Graficar la media (cuadrado rojo)
            plot(pos_SIN, mean(data_SIN), 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
        end
    end
    
    % Crear objetos "dummy" para la leyenda
    h_con = plot(NaN, NaN, 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
    h_sin = plot(NaN, NaN, 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
    legend([h_con, h_sin], {'CON NAFNet','SIN NAFNet'});

    % Agregar anotación adicional que indique el significado de los números debajo de cada modelo
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Los números debajo de cada modelo indican la media de la distancia euclídea con la original (Azul: CON, Rojo: SIN)', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8);
    
    % Si la métrica es sin referencia, añadir debajo del nombre de cada modelo
    % la media de la distancia euclídea (anotada en azul para CON y en rojo para SIN)
    if strcmp(metricName, 'BRISQUE') || strcmp(metricName, 'PIQE')
        y_lim = get(gca, 'YLim');
        offset = 0.05*(y_lim(2)-y_lim(1));
        for model_idx = 1:num_models
            if strcmp(metricName, 'BRISQUE')
                mean_dist_con = mean(metricValues_CON.DIST_BRISQUE{model_idx});
                mean_dist_sin = mean(metricValues_SIN.DIST_BRISQUE{model_idx});
            else
                mean_dist_con = mean(metricValues_CON.DIST_PIQE{model_idx});
                mean_dist_sin = mean(metricValues_SIN.DIST_PIQE{model_idx});
            end
            text(model_idx - 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_con), ...
                'Color', 'b', 'HorizontalAlignment', 'center', 'FontSize', 8);
            text(model_idx + 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_sin), ...
                'Color', 'r', 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    xlabel('Modelos', 'FontSize',12);
    ylabel(metricName, 'FontSize',12);
    title(sprintf('Comparación de %s (CON vs SIN)', metricName), 'FontSize',14);
    xticks(1:num_models);
    xticklabels(model_names);
    grid on;
    hold off;
end
