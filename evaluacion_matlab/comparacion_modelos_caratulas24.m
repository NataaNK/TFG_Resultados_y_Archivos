% Lista de directorios para los datasets procesados
datasets = {
    "E:\MasOrange\Real-ESRGAN\scaled_caratulas24_x4";
    "E:\MasOrange\SwinFIR\results\scaled_caratulas24_x4";
    "E:\MasOrange\HAT\results\scaled_caratulas24_x4";
    "E:\MasOrange\aura-sr\scaled_caratulas_x4";
    "E:\MasOrange\DRCT\natalia\scaled_caratulas_x4";
    "E:\MasOrange\HMA\results\scaled_caratulas24_x4";
    "E:\MasOrange\troncho_portal\StableSR\scaled_caratulas24_x4";
    "E:\MasOrange\Efficient-Computing\LowLevel\IPG\results\scaled_caratulas24_x4"
};

% Directorio de imágenes originales
originals_dir = "E:\MasOrange\caratulas24\caratulas24_hd";

% Métricas a calcular
metrics = {'PSNR', 'SSIM', 'MS-SSIM', 'FSIM'};

% Inicializar matriz de resultados
num_metrics = length(metrics);
num_models = length(datasets);
results = zeros(num_metrics, num_models);

% Índices de imágenes a evaluar
valid_indices = setdiff(1:20, [4, 5, 12, 18]);

% Procesar cada dataset
for model_idx = 1:num_models
    % Obtener el directorio del dataset actual
    dataset_dir = datasets{model_idx};
    
    % Inicializar acumuladores de métricas
    psnr_accum = 0;
    ssim_accum = 0;
    ms_ssim_accum = 0;
    fsim_accum = 0;
    
    % Recorrer índices válidos de imágenes
    for img_idx = valid_indices
        % Cargar imagen original
        original_file = fullfile(originals_dir, sprintf('%d.jpg', img_idx));
        if ~isfile(original_file)
            warning("Imagen original %d.jpg no encontrada en %s", img_idx, originals_dir);
            continue;
        end
        original_img = imread(original_file);
        
        % Buscar imagen procesada correspondiente
        processed_files = dir(fullfile(dataset_dir, '*')); % Obtener todos los archivos del directorio
        matched_file = [];
        
        for i = 1:length(processed_files)
            processed_name = processed_files(i).name;
            
            % Usar regexp para extraer el número inicial del nombre del archivo
            tokens = regexp(processed_name, '^(\d+)', 'tokens');
            if ~isempty(tokens) && str2double(tokens{1}{1}) == img_idx
                matched_file = processed_files(i).name;
                break;
            end
        end
        
        if isempty(matched_file)
            warning("Imagen procesada para %d.jpg no encontrada en %s", img_idx, dataset_dir);
            continue;
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
    num_valid_images = length(valid_indices);
    results(1, model_idx) = psnr_accum / num_valid_images;
    results(2, model_idx) = ssim_accum / num_valid_images;
    results(3, model_idx) = ms_ssim_accum / num_valid_images;
    results(4, model_idx) = fsim_accum / num_valid_images;
end

% Generar tabla para visualización
model_names = {'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};
T = array2table(results, 'RowNames', metrics, 'VariableNames', model_names);

% Mostrar tabla
disp(T);

% Crear imagen para visualización
figure;
uitable('Data', T{:,:}, 'ColumnName', model_names, 'RowName', metrics, ...
    'Position', [20 20 800 200]);

% Guardar tabla de resultados como imagen (opcional)
saveas(gcf, 'ModelComparisonTable.png');
