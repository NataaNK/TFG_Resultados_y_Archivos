% Directorios de los modelos e imágenes originales
directories = {
    "E:\MasOrange\caratulas24\caratulas24_hd"; % Original HD
    "E:\MasOrange\caratulas24\caratulas24_lr_t"; % Low-Res
    "E:\MasOrange\Real-ESRGAN\scaled_caratulas24_x4";
    "E:\MasOrange\SwinFIR\results\scaled_caratulas24_x4";
    "E:\MasOrange\HAT\results\scaled_caratulas24_x4";
    "E:\MasOrange\aura-sr\scaled_caratulas_x4";
    "E:\MasOrange\DRCT\natalia\scaled_caratulas_x4";
    "E:\MasOrange\HMA\results\scaled_caratulas24_x4";
    "E:\MasOrange\troncho_portal\StableSR\scaled_caratulas24_x4";
    "E:\MasOrange\Efficient-Computing\LowLevel\IPG\results\scaled_caratulas24_x4"
};

% Nombres de los modelos
model_names = {'Original HD', 'Low-Res', 'Real-ESRGAN', 'SwinFIR', 'HAT', ...
               'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};

% Índices de imágenes a evaluar (excluyendo 4, 5, 12 y 18)
valid_indices = setdiff(1:20, [4, 5, 12, 18]);

% Crear una figura con pestañas
f = figure('Name', 'Comparación de Modelos - Carátulas', 'NumberTitle', 'off');
tab_group = uitabgroup(f);

% Procesar cada índice de imagen válida
for img_idx = valid_indices
    % Crear una nueva pestaña para esta imagen
    tab = uitab(tab_group, 'Title', sprintf('Imagen %d', img_idx));
    
    % Crear un diseño de cuadrícula para las imágenes
    grid_rows = 2; % Número de filas
    grid_cols = ceil(length(directories) / grid_rows); % Número de columnas
    tiled_layout = tiledlayout(tab, grid_rows, grid_cols, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    
    % Procesar cada directorio (modelos e imágenes originales)
    for model_idx = 1:length(directories)
        % Obtener el directorio del modelo o imágenes originales
        model_dir = directories{model_idx};
        
        % Buscar la imagen correspondiente
        all_files = dir(fullfile(model_dir, '*')); % Obtener todos los archivos del directorio
        matched_file = [];
        
        for i = 1:length(all_files)
            processed_name = all_files(i).name;
            
            % Extraer el número inicial completo del nombre del archivo
            tokens = regexp(processed_name, '^(\d+)', 'tokens');
            if ~isempty(tokens) && str2double(tokens{1}{1}) == img_idx
                matched_file = all_files(i).name;
                break;
            end
        end
        
        if isempty(matched_file)
            fprintf("DEBUG: Imagen no encontrada para índice %d en el directorio %s\n", img_idx, model_dir);
            continue;
        else
            fprintf("DEBUG: Imagen encontrada para índice %d en el directorio %s: %s\n", img_idx, model_dir, matched_file);
        end
        
        % Leer la imagen
        img_path = fullfile(model_dir, matched_file);
        img = imread(img_path);
        
        % Asegurar que la imagen tenga el tamaño correcto
        if model_idx > 2 % Ajustar las imágenes procesadas al tamaño de la original HD
            original_img = imread(fullfile(directories{1}, sprintf('%d.jpg', img_idx)));
            if size(img, 1) ~= size(original_img, 1) || size(img, 2) ~= size(original_img, 2)
                fprintf("DEBUG: Redimensionando imagen %s del modelo %s para que coincida con la original\n", matched_file, model_names{model_idx});
                img = imresize(img, [size(original_img, 1), size(original_img, 2)]);
            end
        end
        
        % Calcular PSNR y SSIM respecto a la imagen original HD
        if model_idx > 2
            psnr_value = psnr(img, original_img);
            ssim_value = ssim(img, original_img);
        else
            psnr_value = NaN;
            ssim_value = NaN;
        end
        
        % Mostrar la imagen en la cuadrícula
        nexttile(tiled_layout);
        imshow(img);
        if model_idx == 1
            title('Original HD');
        elseif model_idx == 2
            title('Low-Res');
        else
            title(sprintf('%s\nPSNR: %.2f, SSIM: %.2f', model_names{model_idx}, psnr_value, ssim_value));
        end
    end
end
