% Lista de directorios para los datasets procesados
datasets = {
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
model_names = {'Real-ESRGAN', 'SwinFIR', 'HAT', 'Aura-SR', 'DRCT', 'HMA', 'StableSR', 'IPG'};

% Inicializar matriz de resultados
num_metrics = 2; % BRISQUE y PIQE
num_models = length(datasets);
results = zeros(num_metrics, num_models);

% Procesar cada dataset
for model_idx = 1:num_models
    % Obtener el directorio del dataset actual
    dataset_dir = datasets{model_idx};
    
    % Obtener lista de imágenes
    all_files = dir(fullfile(dataset_dir, '*.*')); % Ajustar extensión si es necesario
    all_files = all_files(~[all_files.isdir]); % Filtrar directorios
    num_images = length(all_files);
    
    % Inicializar acumuladores de métricas
    brisque_accum = 0;
    piqe_accum = 0;
    
    % Recorrer imágenes
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
        
        % Acumular resultados
        brisque_accum = brisque_accum + brisque_score;
        piqe_accum = piqe_accum + piqe_score;
        
        % Mostrar depuración
        fprintf('DEBUG: Procesando imagen %s - BRISQUE: %.2f, PIQE: %.2f\n', ...
                img_file, brisque_score, piqe_score);
    end
    
    % Calcular promedios para el dataset
    results(1, model_idx) = brisque_accum / num_images;
    results(2, model_idx) = piqe_accum / num_images;
end

% Generar tabla para visualización
metrics = {'BRISQUE', 'PIQE'};
T = array2table(results, 'RowNames', metrics, 'VariableNames', model_names);

% Mostrar tabla
disp(T);

% Crear imagen para visualización
figure;
uitable('Data', T{:,:}, 'ColumnName', model_names, 'RowName', metrics, ...
    'Position', [20 20 800 200]);

% Guardar tabla de resultados como imagen (opcional)
saveas(gcf, 'ModelComparisonTable_NoReference_Brisque_Piqe.png');
