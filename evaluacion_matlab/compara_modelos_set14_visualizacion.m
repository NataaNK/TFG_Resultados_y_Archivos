% Lista de directorios para los datasets procesados
datasets = {
    "E:\MasOrange\originales\poster_downloads_44"; % Low-Resolution (LR)
    "E:\MasOrange\Real-ESRGAN\scaled_posters_x4";
    "E:\MasOrange\SwinFIR\results\scaled_posters44_x4";
    "E:\MasOrange\HAT\results\scaled_posters_x4";
    "E:\MasOrange\aura-sr\scaled_posters_x4";
    "E:\MasOrange\DRCT\natalia\scaled_posters_x4";
    "E:\MasOrange\HMA\results\scaled_posters44_x4";
    "E:\MasOrange\troncho_portal\StableSR\scaled_posters44_x4";
    "E:\MasOrange\Efficient-Computing\LowLevel\IPG\results\scaled_posters44_x4"
};

% Nombres de los modelos
model_names = {'Low-Res (LR)', 'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};

% Procesar cada dataset
figure('Name', 'Visualización de Imágenes y Métricas', 'NumberTitle', 'off');
tiled_layout = tiledlayout(length(datasets), 44, 'TileSpacing', 'compact', 'Padding', 'compact');

for model_idx = 1:length(datasets)
    % Obtener el directorio del dataset actual
    dataset_dir = datasets{model_idx};
    
    % Obtener lista de imágenes
    all_files = dir(fullfile(dataset_dir, '*.*')); % Ajustar extensión si es necesario
    all_files = all_files(~[all_files.isdir]); % Filtrar directorios
    num_images = length(all_files);
    
    for img_idx = 1:num_images
        % Cargar imagen
        img_file = all_files(img_idx).name;
        img_path = fullfile(dataset_dir, img_file);
        img = imread(img_path);
        
        % Convertir a escala de grises si es necesario
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Calcular métricas sin referencia
        brisque_score = brisque(img);
        piqe_score = piqe(img);
        
        % Visualizar imagen y métricas
        nexttile;
        imshow(img);
        title(sprintf('%s\nBRISQUE: %.2f\nPIQE: %.2f', model_names{model_idx}, brisque_score, piqe_score), ...
              'FontSize', 8, 'Interpreter', 'none');
        
        % Mostrar depuración
        fprintf('DEBUG: Modelo %s, Imagen %s - BRISQUE: %.2f, PIQE: %.2f\n', ...
                model_names{model_idx}, img_file, brisque_score, piqe_score);
    end
end

% Ajustar el diseño
title(tiled_layout, 'Comparación de Imágenes y Métricas (BRISQUE y PIQE)', 'FontSize', 14);
