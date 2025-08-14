% Directorios de los datasets
datasets = {
    "E:\MasOrange\originales\poster_downloads_44"; % LR
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
model_names = {'LR', 'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};

% Obtener lista de imágenes en el directorio LR
lr_files = dir(fullfile(datasets{1}, '*.*'));
lr_files = lr_files(~[lr_files.isdir]); % Filtrar directorios
num_images = length(lr_files);

% Crear figura con pestañas
f = figure('Name', 'Comparación de Imágenes - BRISQUE y PIQE', 'NumberTitle', 'off');
tab_group = uitabgroup(f);

% Recorrer cada imagen
for img_idx = 1:num_images
    % Crear pestaña para la imagen actual
    tab = uitab(tab_group, 'Title', sprintf('Imagen %d', img_idx));
    tiled_layout = tiledlayout(tab, 2, ceil(length(datasets) / 2), 'TileSpacing', 'Compact', 'Padding', 'Compact');
    
    % Recorrer cada modelo (incluyendo LR)
    for model_idx = 1:length(datasets)
        % Cargar imagen del modelo actual
        model_dir = datasets{model_idx};
        all_files = dir(fullfile(model_dir, '*.*'));
        all_files = all_files(~[all_files.isdir]); % Filtrar directorios
        
        if img_idx <= length(all_files)
            img_file = all_files(img_idx).name;
            img_path = fullfile(model_dir, img_file);
            img = imread(img_path);
        else
            warning('Imagen %d no encontrada en %s', img_idx, model_dir);
            continue;
        end
        
        % Calcular métricas BRISQUE y PIQE
        brisque_score = brisque(img);
        piqe_score = piqe(img);
        
        % Mostrar la imagen en la pestaña
        nexttile(tiled_layout);
        imshow(img);
        title(sprintf('%s\nBRISQUE: %.2f, PIQE: %.2f', model_names{model_idx}, brisque_score, piqe_score), ...
              'Interpreter', 'none', 'FontSize', 8);
    end
end
