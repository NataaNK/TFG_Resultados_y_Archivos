%% Comparación de resultados: X vs Y
% Definir constantes para la comparación
nameX = 'x4v3 ORIGINAL INTERPOLACIÓN';  % Condición X
nameY = 'OPTUNA';  % Condición Y

% Definir el nombre del modelo (único) a comparar, por ejemplo "RealESRGAN"
model_name = 'RealESRGAN';

%% Gráfico comparativo de tiempos de ejecución
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
subdataset_template_X = {... 
    "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\x4v3wdn_default"};
ref_dirs_X = {... 
   "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\FinetuningDataset\ConTexto100Original"};

% Inicializar estructura sin BRISQUE
metricValues_X.PSNR    = [];
metricValues_X.SSIM    = [];
metricValues_X.MS_SSIM = [];
metricValues_X.FSIM    = [];
metricValues_X.PIQE    = [];
metricValues_X.DIST_PIQE    = [];

for sub_idx = 1:length(subdataset_template_X)
    current_dataset = subdataset_template_X{sub_idx};
    current_ref_dir = ref_dirs_X{sub_idx};
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
            % Solo PIQE sin referencia
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular PIQE para %s.', processed_name);
            end
            continue;
        end

        img_index = str2double(tokens{1}{1});
        if contains(processed_name, 'BN')
            keyword = 'BN';
        elseif contains(processed_name, 'Color')
            keyword = 'Color';
        elseif contains(processed_name, 'Sepia')
            keyword = 'Sepia';
        else
            warning('No se encontró palabra clave en %s', processed_name);
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular PIQE para %s.', processed_name);
            end
            continue;
        end

        % Buscar referencia
        ref_files = dir(fullfile(current_ref_dir, '*.*'));
        pattern = sprintf('^%d_%s.*', img_index, keyword);
        matches = arrayfun(@(x) ~isempty(regexp(x.name, pattern, 'once')), ref_files);
        ref_info = ref_files(matches);
        if isempty(ref_info)
            warning('No se encontró la imagen de referencia para %d con keyword %s', img_index, keyword);
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular PIQE para %s.', processed_name);
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
                piqe_score    = piqe(processed_gray);
                metricValues_X.PIQE    = [metricValues_X.PIQE, piqe_score];
            catch
                warning('No se pudo calcular PIQE para %s.', processed_name);
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
            processed_gray = imresize(processed_gray, size(original_gray));
        end

        % Métricas con referencia
        try
            metricValues_X.PSNR = [metricValues_X.PSNR, psnr(processed_gray, original_gray)];
        catch; warning('PSNR falló para %s', processed_name); end
        try
            metricValues_X.SSIM = [metricValues_X.SSIM, ssim(processed_gray, original_gray)];
        catch; warning('SSIM falló para %s', processed_name); end
        if exist('multissim','file')
            try
                metricValues_X.MS_SSIM = [metricValues_X.MS_SSIM, multissim(processed_gray, original_gray)];
            catch; warning('MS-SSIM falló para %s', processed_name); end
        end
        try
            metricValues_X.FSIM = [metricValues_X.FSIM, FSIM(processed_gray, original_gray)];
        catch; warning('FSIM falló para %s', processed_name); end

        % PIQE y distancia PIQE
        try
            piqe_original = piqe(original_gray);
        catch
            warning('PIQE original falló para %s', processed_name);
            piqe_original = NaN;
        end
        try
            piqe_score = piqe(processed_gray);
            metricValues_X.PIQE = [metricValues_X.PIQE, piqe_score];
            if ~isnan(piqe_original)
                metricValues_X.DIST_PIQE = [metricValues_X.DIST_PIQE, abs(piqe_score - piqe_original)];
            end
            catch0
            warning('PIQE falló para %s', processed_name);
        end
    end
end

%% Procesamiento de métricas para Y 
subdataset_template_Y = {... 
    "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\4_1_piqe_registro_optuna_ConTexto100\mejor_salida_optuna_contexto100"};
ref_dirs_Y = {... 
   "C:\Users\nekos\OneDrive\Escritorio\SalidasConTexto100\FinetuningDataset\ConTexto100Original"};

metricValues_Y.PSNR    = [];
metricValues_Y.SSIM    = [];
metricValues_Y.MS_SSIM = [];
metricValues_Y.FSIM    = [];
metricValues_Y.PIQE    = [];
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
            warning('Índice no extraído: %s', processed_name);
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
            catch
                warning('PIQE falló para %s.', processed_name);
            end
            continue;
        end

        img_index = str2double(tokens{1}{1});
        if contains(processed_name, 'BN')
            keyword = 'BN';
        elseif contains(processed_name, 'Color')
            keyword = 'Color';
        elseif contains(processed_name, 'Sepia')
            keyword = 'Sepia';
        else
            keyword = '';
        end

        if ~isempty(keyword)
            ref_pattern = fullfile(current_ref_dir, sprintf('%d_%s.*', img_index, keyword));
        else
            ref_pattern = fullfile(current_ref_dir, sprintf('%d.*', img_index));
        end
        ref_info = dir(ref_pattern);
        if isempty(ref_info)
            warning('Referencia no encontrada para %d', img_index);
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
            catch
                warning('PIQE falló para %s.', processed_name);
            end
            continue;
        else
            ref_file = fullfile(current_ref_dir, ref_info(1).name);
        end

        try
            original_img = imread(ref_file);
        catch
            warning('No se pudo leer referencia: %s', ref_file);
            if size(processed_img,3)==3
                processed_gray = rgb2gray(processed_img);
            else
                processed_gray = processed_img;
            end
            try
                piqe_score    = piqe(processed_gray);
                metricValues_Y.PIQE    = [metricValues_Y.PIQE, piqe_score];
            catch
                warning('PIQE falló para %s.', processed_name);
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
            processed_gray = imresize(processed_gray, size(original_gray));
        end

        try
            metricValues_Y.PSNR = [metricValues_Y.PSNR, psnr(processed_gray, original_gray)];
        catch; warning('PSNR falló para %s', processed_name); end
        try
            metricValues_Y.SSIM = [metricValues_Y.SSIM, ssim(processed_gray, original_gray)];
        catch; warning('SSIM falló para %s', processed_name); end
        if exist('multissim','file')
            try
                metricValues_Y.MS_SSIM = [metricValues_Y.MS_SSIM, multissim(processed_gray, original_gray)];
            catch; warning('MS-SSIM falló para %s', processed_name); end
        end
        try
            metricValues_Y.FSIM = [metricValues_Y.FSIM, FSIM(processed_gray, original_gray)];
        catch; warning('FSIM falló para %s', processed_name); end

        try
            piqe_original = piqe(original_gray);
        catch
            warning('PIQE original falló para %s', processed_name);
            piqe_original = NaN;
        end
        try
            piqe_score = piqe(processed_gray);
            metricValues_Y.PIQE = [metricValues_Y.PIQE, piqe_score];
            if ~isnan(piqe_original)
                metricValues_Y.DIST_PIQE = [metricValues_Y.DIST_PIQE, abs(piqe_score - piqe_original)];
            end
        catch
            warning('PIQE falló para %s', processed_name);
        end
    end
end

%% Gráficos combinados de métricas (X vs Y)
metricsList = {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM', 'PIQE'};  % Se quitó 'BRISQUE'
for m = 1:length(metricsList)
    metricName = metricsList{m};
    figure; hold on;
    pos_X = 1 - 0.15;
    pos_Y = 1 + 0.15;
    data_X = metricValues_X.(metricName);
    data_Y = metricValues_Y.(metricName);

    if ~isempty(data_X)
        boxchart(repmat(pos_X, size(data_X)), data_X, 'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'b');
        plot(pos_X, mean(data_X), 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
    end
    if ~isempty(data_Y)
        boxchart(repmat(pos_Y, size(data_Y)), data_Y, 'BoxWidth', 0.1, 'MarkerStyle', 'none', 'BoxFaceColor', 'r');
        plot(pos_Y, mean(data_Y), 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
    end

    h_X = plot(NaN, NaN, 'bs', 'MarkerSize',8, 'MarkerFaceColor','b');
    h_Y = plot(NaN, NaN, 'rs', 'MarkerSize',8, 'MarkerFaceColor','r');
    legend([h_X, h_Y], {nameX, nameY});

    % Anotación de distancia para PIQE únicamente
    if strcmp(metricName, 'PIQE')
        y_lim = get(gca, 'YLim');
        offset = 0.05*(y_lim(2)-y_lim(1));
        mean_dist_X = mean(metricValues_X.DIST_PIQE);
        mean_dist_Y = mean(metricValues_Y.DIST_PIQE);
        text(1 - 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_X), 'Color', 'b', 'HorizontalAlignment', 'center', 'FontSize', 8);
        text(1 + 0.1, y_lim(1)-offset, sprintf('%.2f', mean_dist_Y), 'Color', 'r', 'HorizontalAlignment', 'center', 'FontSize', 8);
    end

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

%% Cálculo y despliegue de mejoras
overall_weighted_sum = 0;
total_weight = 0;

for m = 1:length(metricsList)
    metricName = metricsList{m};
    data_X = metricValues_X.(metricName);
    data_Y = metricValues_Y.(metricName);

    mean_X = mean(data_X);
    mean_Y = mean(data_Y);
    std_X = std(data_X);
    std_Y = std(data_Y);

    if ismember(metricName, {'PSNR', 'SSIM', 'MS_SSIM', 'FSIM'})
        base_improvement = ((mean_Y - mean_X) / mean_X) * 100;
    else  % PIQE
        base_improvement = ((mean_X - mean_Y) / mean_X) * 100;
    end

    weight = 1 / (((std_X + std_Y) / 2) + eps);
    weighted_improvement = base_improvement * weight;

    overall_weighted_sum = overall_weighted_sum + weighted_improvement;
    total_weight = total_weight + weight;

    fprintf('Métrica %s:\n  Media X = %.3f, std X = %.3f\n  Media Y = %.3f, std Y = %.3f\n  Porcentaje de mejora (ponderada) de Y respecto a X: %.2f%%\n\n', ...
        metricName, mean_X, std_X, mean_Y, std_Y, weighted_improvement);
end

overall_improvement = overall_weighted_sum / total_weight;
fprintf('Mejora global (ponderada) considerando todas las métricas: %.2f%%\n', overall_improvement);
