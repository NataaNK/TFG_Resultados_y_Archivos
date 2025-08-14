%% Comparación de resultados: X vs Y
% Definir constantes para la comparación
nameX = 'x4v3 ORIGINAL INTERPOLACIÓN';  % Condición X
nameY = 'LoRA L1 + VGG19 + GAN (EPOCH 6)';  % Condición Y

% Definir el nombre del modelo (único) a comparar, por ejemplo "RealESRGAN"
model_name = 'RealESRGAN';

%% Gráfico comparativo de tiempos de ejecución
% Se emplean tiempos de ejemplo para la condición X y Y
timeX_sec = 1138.27 + 708.01;  % Ejemplo para X
timeY_sec = 733.62;            % Ejemplo para Y 
timeX_min = timeX_sec / 60;
timeY_min = timeY_sec / 60;

combined_times = [timeX_min, timeY_min];
figure;
bar(combined_times);
set(gca, 'XTick', 1:2, 'XTickLabel', {nameX, nameY}, 'FontSize', 12);
ylabel('Tiempo (minutos)', 'FontSize', 12);
title('Comparación de Tiempos de Ejecución', 'FontSize', 14);
grid on;

%% Procesamiento de métricas para X 
% Rutas para la condición X
subdataset_template_X = {...
    "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\x4v3wdn_default"};
ref_dirs_X = {...
   "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\FinetuningDataset\ConTexto100Original"};

% Inicializar estructura para almacenar las métricas (CON -> X)
metricValues_X.PSNR    = [];
metricValues_X.SSIM    = [];
metricValues_X.MS_SSIM = [];
metricValues_X.FSIM    = [];
metricValues_X.BRISQUE = [];
metricValues_X.PIQE    = [];
% Nuevos campos para distancias euclídeas (sin referencia)
metricValues_X.DIST_BRISQUE = [];
metricValues_X.DIST_PIQE    = [];

for sub_idx = 1:length(subdataset_template_X)
    current_dataset = subdataset_template_X{sub_idx};
    current_ref_dir = ref_dirs_X{sub_idx};
    
    % Obtener lista de imágenes (omitiendo carpetas)
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
        
        % EXTRAER EL ÍNDICE (hasta 2 dígitos) DEL NOMBRE DEL ARCHIVO
        tokens = regexp(processed_name, '^(\d{1,2})', 'tokens');
        if isempty(tokens)
            warning('No se pudo extraer el índice de la imagen: %s', processed_name);
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                metricValues_X.BRISQUE = [metricValues_X.BRISQUE, brisque_score];
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        end
        
        img_index = str2double(tokens{1}{1});
        
        %% NUEVA LÓGICA PARA BUSCAR LA IMAGEN DE REFERENCIA (CON X)
        if contains(processed_name, 'BN')
            keyword = 'BN';
        elseif contains(processed_name, 'Color')
            keyword = 'Color';
        elseif contains(processed_name, 'Sepia')
            keyword = 'Sepia';
        else
            warning('No se encontró palabra clave (BN, Color o Sepia) en %s', processed_name);
            if size(processed_img, 3) == 3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                brisque_score = brisque(processed_gray);
                piqe_score    = piqe(processed_gray);
                metricValues_X.BRISQUE = [metricValues_X.BRISQUE, brisque_score];
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        end
        
        % Buscar la imagen de referencia en current_ref_dir usando la palabra clave
        ref_files = dir(fullfile(current_ref_dir, '*.*'));
        pattern = sprintf('^%d_%s.*', img_index, keyword);
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
                metricValues_X.BRISQUE = [metricValues_X.BRISQUE, brisque_score];
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        else
            ref_file = fullfile(current_ref_dir, ref_info(1).name);
        end
        %% FIN NUEVA LÓGICA
        
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
                metricValues_X.BRISQUE = [metricValues_X.BRISQUE, brisque_score];
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
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
            metricValues_X.PSNR = [metricValues_X.PSNR, psnr_val];
        catch
            warning('Error al calcular PSNR para %s.', processed_name);
        end
        try
            ssim_val = ssim(processed_gray, original_gray);
            metricValues_X.SSIM = [metricValues_X.SSIM, ssim_val];
        catch
            warning('Error al calcular SSIM para %s.', processed_name);
        end
        if exist('multissim', 'file')
            try
                ms_ssim_val = multissim(processed_gray, original_gray);
                metricValues_X.MS_SSIM = [metricValues_X.MS_SSIM, ms_ssim_val];
            catch
                warning('Error al calcular MS-SSIM para %s.', processed_name);
            end
        end
        try
            fsim_val = FSIM(processed_gray, original_gray);
            metricValues_X.FSIM = [metricValues_X.FSIM, fsim_val];
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
            metricValues_X.BRISQUE = [metricValues_X.BRISQUE, brisque_score];
            if ~isnan(brisque_original)
                dist_br = abs(brisque_score - brisque_original);
                metricValues_X.DIST_BRISQUE = [metricValues_X.DIST_BRISQUE, dist_br];
            end
        catch
            warning('Error al calcular BRISQUE para %s.', processed_name);
        end
        
        try
            piqe_score = piqe(processed_gray);
            metricValues_X.PIQE = [metricValues_X.PIQE, piqe_score];
            if ~isnan(piqe_original)
                dist_piqe = abs(piqe_score - piqe_original);
                metricValues_X.DIST_PIQE = [metricValues_X.DIST_PIQE, dist_piqe];
            end
        catch
            warning('Error al calcular PIQE para %s.', processed_name);
        end
    end
end

%% Procesamiento de métricas para Y 
% Rutas para la condición Y 
subdataset_template_Y = {...
    "C:\Users\nekos\OneDrive\Escritorio\LoRA\L1_VGG_GAN\LoRA_epoch6_ConTexto100"};
ref_dirs_Y = {...
   "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\FinetuningDataset\ConTexto100Original"};

% Inicializar estructura para almacenar las métricas (SIN -> Y)
metricValues_Y.PSNR    = [];
metricValues_Y.SSIM    = [];
metricValues_Y.MS_SSIM = [];
metricValues_Y.FSIM    = [];
metricValues_Y.BRISQUE = [];
metricValues_Y.PIQE    = [];
metricValues_Y.DIST_BRISQUE = [];
metricValues_Y.DIST_PIQE    = [];

for sub_idx = 1:length(subdataset_template_Y)
    current_dataset = subdataset_template_Y{sub_idx};
    current_ref_dir = ref_dirs_Y{sub_idx};
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
                metricValues_Y.BRISQUE = [metricValues_Y.BRISQUE, brisque_score];
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        end
        
        img_index = str2double(tokens{1}{1});
        
        %% NUEVA LÓGICA PARA BUSCAR LA IMAGEN DE REFERENCIA (CON Y)
        if contains(processed_name, 'BN')
            keyword = 'BN';
        elseif contains(processed_name, 'Color')
            keyword = 'Color';
        elseif contains(processed_name, 'Sepia')
            keyword = 'Sepia';
        else
            warning('No se encontró palabra clave (BN, Color o Sepia) en %s', processed_name);
            % Se continúa sin palabra clave, usando la búsqueda por índice solamente
            keyword = '';
        end
        
        if ~isempty(keyword)
            ref_pattern = fullfile(current_ref_dir, sprintf('%d_%s.*', img_index, keyword));
        else
            ref_pattern = fullfile(current_ref_dir, sprintf('%d.*', img_index));
        end
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
                metricValues_Y.BRISQUE = [metricValues_Y.BRISQUE, brisque_score];
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
            catch
                warning('No se pudo calcular BRISQUE/PIQE para %s.', processed_name);
            end
            continue;
        else
            ref_file = fullfile(current_ref_dir, ref_info(1).name);
        end
        %% FIN NUEVA LÓGICA
        
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
                metricValues_Y.BRISQUE = [metricValues_Y.BRISQUE, brisque_score];
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
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
            metricValues_Y.PSNR = [metricValues_Y.PSNR, psnr_val];
        catch
            warning('Error al calcular PSNR para %s.', processed_name);
        end
        try
            ssim_val = ssim(processed_gray, original_gray);
            metricValues_Y.SSIM = [metricValues_Y.SSIM, ssim_val];
        catch
            warning('Error al calcular SSIM para %s.', processed_name);
        end
        if exist('multissim','file')
            try
                ms_ssim_val = multissim(processed_gray, original_gray);
                metricValues_Y.MS_SSIM = [metricValues_Y.MS_SSIM, ms_ssim_val];
            catch
                warning('Error al calcular MS-SSIM para %s.', processed_name);
            end
        end
        try
            fsim_val = FSIM(processed_gray, original_gray);
            metricValues_Y.FSIM = [metricValues_Y.FSIM, fsim_val];
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
            metricValues_Y.BRISQUE = [metricValues_Y.BRISQUE, brisque_score];
            if ~isnan(brisque_original)
                dist_br = abs(brisque_score - brisque_original);
                metricValues_Y.DIST_BRISQUE = [metricValues_Y.DIST_BRISQUE, dist_br];
            end
        catch
            warning('Error al calcular BRISQUE para %s.', processed_name);
        end
        
        try
            piqe_score = piqe(processed_gray);
            metricValues_Y.PIQE = [metricValues_Y.PIQE, piqe_score];
            if ~isnan(piqe_original)
                dist_piqe = abs(piqe_score - piqe_original);
                metricValues_Y.DIST_PIQE = [metricValues_Y.DIST_PIQE, dist_piqe];
            end
        catch
            warning('Error al calcular PIQE para %s.', processed_name);
        end
    end
end

%% Gráficos combinados de métricas (X vs Y)
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'BRISQUE', 'PIQE'};
for m = 1:length(metricsList)
    metricName = metricsList{m};
    figure; hold on;
    
    % Para un único grupo se usan posiciones X con un leve desplazamiento:
    pos_X = 1 - 0.15;
    pos_Y = 1 + 0.15;
    
    data_X = metricValues_X.(metricName);
    data_Y = metricValues_Y.(metricName);
    
    if ~isempty(data_X)
        boxchart(repmat(pos_X, size(data_X)), data_X, ...
            'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'b');
        % Graficar la media (cuadrado azul)
        plot(pos_X, mean(data_X), 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
    end
    if ~isempty(data_Y)
        boxchart(repmat(pos_Y, size(data_Y)), data_Y, ...
            'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'r');
        % Graficar la media (cuadrado rojo)
        plot(pos_Y, mean(data_Y), 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
    end
    
    % Objetos "dummy" para la leyenda
    h_X = plot(NaN, NaN, 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
    h_Y = plot(NaN, NaN, 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
    legend([h_X, h_Y], {nameX, nameY});
    
    % Anotación que indica la media de la distancia euclídea con la original
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', sprintf('Los números debajo del modelo indican la media de la distancia euclídea con la original (Azul: %s, Rojo: %s)', nameX, nameY), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8);
    
    % Si la métrica es sin referencia, añadir debajo la media de la distancia euclídea
    if strcmp(metricName, 'BRISQUE') || strcmp(metricName, 'PIQE')
        y_lim = get(gca, 'YLim');
        offset = 0.05*(y_lim(2)-y_lim(1));
        if strcmp(metricName, 'BRISQUE')
            mean_dist_X = mean(metricValues_X.DIST_BRISQUE);
            mean_dist_Y = mean(metricValues_Y.DIST_BRISQUE);
        else
            mean_dist_X = mean(metricValues_X.DIST_PIQE);
            mean_dist_Y = mean(metricValues_Y.DIST_PIQE);
        end
        text(1 - 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_X), ...
            'Color', 'b', 'HorizontalAlignment', 'center', 'FontSize', 8);
        text(1 + 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_Y), ...
            'Color', 'r', 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    
    % Escapar el guion bajo para que se muestre correctamente MS_SSIM
    if strcmp(metricName, 'MS_SSIM')
        metricTitle = 'MS\_SSIM';
        yLabelText = 'MS\_SSIM';
    else
        metricTitle = metricName;
        yLabelText = metricName;
    end
    xlabel('Modelo', 'FontSize',12);
    ylabel(yLabelText, 'FontSize',12);
    title(sprintf('Comparación de %s (%s vs %s)', metricTitle, nameX, nameY), 'FontSize',14);
    xticks(1);
    xticklabels({model_name});
    grid on;
    hold off;
end
