% Lista de directorios para los datasets procesados
datasets = {
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\nearest\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\bicubic\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\Real-ESRGAN\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\SwinFIR\results\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\HAT\results\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\aura-sr\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\DRCT\natalia\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\HMA\results\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\troncho_portal\StableSR\scaled_set14_x4";
    "C:\Users\nekos\OneDrive\Escritorio\MasOrange\Efficient-Computing\LowLevel\IPG\results\scaled_set14_x4"
};

% Directorio de imágenes originales
originals_dir = "C:\Users\nekos\OneDrive\Escritorio\MasOrange\originales\Set14_HR";

% Métricas a calcular
metrics = {'PSNR', 'SSIM', 'MS-SSIM', 'FSIM'};

% Inicializar matriz de resultados
num_metrics = length(metrics);
num_models = length(datasets);
results = zeros(num_metrics, num_models);

% Procesar cada dataset
for model_idx = 1:num_models
    % Obtener el directorio del dataset actual
    dataset_dir = datasets{model_idx};
    
    % Obtener lista de imágenes originales
    original_files = dir(fullfile(originals_dir, '*.png')); % Ajustar extensión si es necesario
    num_images = length(original_files);
    
    % Inicializar acumuladores de métricas
    psnr_accum = 0;
    ssim_accum = 0;
    ms_ssim_accum = 0;
    fsim_accum = 0;
    
    % Recorrer imágenes originales
    for img_idx = 1:num_images
        % Cargar imagen original
        original_file = original_files(img_idx).name;
        original_img = imread(fullfile(originals_dir, original_file));
        
        % Buscar imagen procesada correspondiente
        processed_files = dir(fullfile(dataset_dir, '*'));
        matched_file = [];
        
        for i = 1:length(processed_files)
            processed_name = processed_files(i).name;
            
            % Extraer el número inicial completo del nombre del archivo
            tokens = regexp(processed_name, '^(\d+)', 'tokens');
            if ~isempty(tokens) && str2double(tokens{1}{1}) == img_idx
                matched_file = processed_files(i).name;
                break;
            end
        end
        
        if isempty(matched_file)
            warning("Imagen procesada para %s no encontrada en %s", original_file, dataset_dir);
            continue;
        else
            fprintf("DEBUG: Imagen encontrada para índice %d en %s: %s\n", img_idx, dataset_dir, matched_file);
        end
        
        % Cargar imagen procesada
        processed_img = imread(fullfile(dataset_dir, matched_file));
        
        % Ajustar el tamaño de la imagen procesada para que coincida con la original
        if size(processed_img, 1) ~= size(original_img, 1) || size(processed_img, 2) ~= size(original_img, 2)
            processed_img = imresize(processed_img, [size(original_img, 1), size(original_img, 2)]);
        end
        
        % Convertir a escala de grises si es necesario
        if size(original_img, 3) == 3
            original_img = rgb2gray(original_img);
        end
        if size(processed_img, 3) == 3
            processed_img = rgb2gray(processed_img);
        end
        
        % Calcular métricas
        psnr_accum = psnr_accum + psnr(processed_img, original_img);
        ssim_accum = ssim_accum + ssim(processed_img, original_img);
        if exist('multissim', 'file')
            ms_ssim_accum = ms_ssim_accum + multissim(processed_img, original_img);
        else
            warning("La función multissim no está disponible en tu MATLAB.");
        end
        fsim_accum = fsim_accum + FSIM(processed_img, original_img);
    end
    
    % Calcular promedios para el dataset
    results(1, model_idx) = psnr_accum / num_images;
    results(2, model_idx) = ssim_accum / num_images;
    results(3, model_idx) = ms_ssim_accum / num_images;
    results(4, model_idx) = fsim_accum / num_images;
end

% Generar tabla para visualización
model_names = {'Nearest', 'Bicubic', 'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};
T = array2table(results, 'RowNames', metrics, 'VariableNames', model_names);

% Mostrar tabla
disp(T);

% Crear imagen para visualización
figure;
uitable('Data', T{:,:}, 'ColumnName', model_names, 'RowName', metrics, ...
    'Position', [20 20 800 200]);

% Guardar tabla de resultados como imagen (opcional)
saveas(gcf, 'ModelComparisonTable.png');
