function calcular_metricas(original_path, procesada_path)
    % Leer imágenes
    try
        original_img = imread(original_path);
        processed_img = imread(procesada_path);
    catch ME
        error("Error al cargar las imágenes: %s", ME.message);
    end
    
    % Ajustar tamaño de la imagen procesada para que coincida con la original
    if size(original_img, 1) ~= size(processed_img, 1) || size(original_img, 2) ~= size(processed_img, 2)
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
    psnr_value = psnr(processed_img, original_img);
    ssim_value = ssim(processed_img, original_img);
    fsim_value = FSIM(processed_img, original_img);
    
    % MS-SSIM (verificar si la función multissim está disponible)
    if exist('multissim', 'file')
        ms_ssim_value = multissim(processed_img, original_img);
    else
        warning("La función multissim no está disponible en tu MATLAB.");
        ms_ssim_value = NaN; % Valor no disponible
    end
    
    % Mostrar resultados
    fprintf('Resultados de las métricas:\n');
    fprintf('PSNR: %.2f\n', psnr_value);
    fprintf('SSIM: %.2f\n', ssim_value);
    fprintf('MS-SSIM: %.2f\n', ms_ssim_value);
    fprintf('FSIM: %.2f\n', fsim_value);
end
