function compara_modelos_caratulas100_visualizacion()

    % ------------------------------------------------------
    % Directorios principales
    % ------------------------------------------------------
    original_dir_resized   = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals";          
    original_dir_deformed  = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals_deformaciones"; 
    lr_dir                 = "C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120";  % Donde están las LR / Deformed
    
    % Directorios de modelos SR (caratulas100)
    model_dirs = {
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\RealESRGAN_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\SWINFIR_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\HAT_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\AuraSR_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\DRTC_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\HMA_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\StableSR_caratulas100"
        "C:\Users\nekos\OneDrive\Escritorio\troncho_portal\troncho_portal\IPG_caratulas100"
    };

    % Nombres de los modelos (en el mismo orden de 'model_dirs')
    model_names = {
        'RealESRGAN'
        'SwinFIR'
        'HAT'
        'AuraSR'
        'DRTC'
        'HMA'
        'StableSR'
        'IPG'
    };

    % ------------------------------------------------------
    % 1) Lista de imágenes LR/Deformed
    % ------------------------------------------------------
    lr_files = dir(fullfile(lr_dir, '*.jpg')); 
    % si tienes también .png, etc., ajusta aquí

    % ------------------------------------------------------
    % 2) Crear la figura con pestañas
    % ------------------------------------------------------
    f = figure('Name', 'Comparación de Modelos - Carátulas120', ...
               'NumberTitle','off', 'WindowState','maximized');  
    tab_group = uitabgroup(f);

    % ------------------------------------------------------
    % 3) Procesar cada imagen LR/Deformed
    % ------------------------------------------------------
    for k = 1:length(lr_files)
        
        % Nombre de la LR
        lr_name = lr_files(k).name;  % p.ej. "1_blur.jpg", "1_resized.jpg"
        lr_path = fullfile(lr_dir, lr_name);

        % Extraer el número y el sufijo
        % Permitimos cualquier número de dígitos con '^(\d+)_(\w+)\.'
        tokens = regexp(lr_name, '^(\d+)_(\w+)\.', 'tokens');
        if isempty(tokens)
            warning('El archivo "%s" no sigue el patrón "[num]_[deform].jpg". Se omite.', lr_name);
            continue;
        end
        idx_str    = tokens{1}{1};   % "1", "2", "100", etc.
        deform_str = tokens{1}{2};   % "blur", "compression", "resized", etc.

        img_index  = str2double(idx_str);

        % --------------------------------------------------
        % 4) Determinar la referencia
        %    - si es "resized" => C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals\<idx>.jpg
        %    - si no => C:\Users\nekos\OneDrive\Escritorio\Caratulas120_originals_deform_las_3\Caratulas120_originals_deformaciones\<idx>_downscaled.jpg
        % --------------------------------------------------
        if contains(deform_str, 'resized', 'IgnoreCase',true)
            ref_file      = fullfile(original_dir_resized, sprintf('%d.jpg', img_index));
            ref_title_str = sprintf('Original (resized)');
        else
            ref_file      = fullfile(original_dir_deformed, sprintf('%d_downscaled.jpg', img_index));
            ref_title_str = sprintf('Original (downscaled)');
        end

        has_reference = false;
        if isfile(ref_file)
            try
                ref_img = imread(ref_file);
                has_reference = true;
            catch
                warning('No se pudo leer la imagen de referencia: %s', ref_file);
            end
        else
            warning('No se encontró la referencia: %s', ref_file);
        end

        % Crear la pestaña
        tab_title = sprintf('Img %d (%s)', img_index, deform_str);
        tab       = uitab(tab_group, 'Title', tab_title);

        % Layout 3 filas x 4 columnas => 12 celdas
        tiled_layout = tiledlayout(tab, 3, 4, 'TileSpacing','loose','Padding','loose');
        
        % --------------------------------------------------
        % Tile 1: Original
        % --------------------------------------------------
        nexttile(tiled_layout);
        if has_reference
            imshow(ref_img);
            title(ref_title_str, 'Interpreter','none');
        else
            imshow(uint8(ones(200,200,3)*128)); % Gris si no existe
            title(sprintf('%s\n(Referencia no encontrada)', ref_title_str), 'Interpreter','none');
        end
        
        % --------------------------------------------------
        % Tile 2: LR/Deformed
        % --------------------------------------------------
        nexttile(tiled_layout);
        try
            lr_img_data = imread(lr_path);
            imshow(lr_img_data);
        catch
            imshow(uint8(ones(200,200,3)*128));
            title(sprintf('LR/Deformed (%s)\nError al leer', deform_str), 'Interpreter','none');
            continue;
        end

        % Métricas
        [psnr_lr, ssim_lr, brisque_lr, piqe_lr] = deal(NaN);

        if has_reference
            try
                % Convertir a gris (opcional, si deseas comparar en 1 canal)
                ref_gray = ref_img;
                if size(ref_gray,3)==3
                    ref_gray = rgb2gray(ref_gray);
                end
                
                proc_lr = lr_img_data;
                if size(proc_lr,3)==3
                    proc_lr = rgb2gray(proc_lr);
                end

                % Ajustar tamaño si difiere
                if size(proc_lr,1)~=size(ref_gray,1) || size(proc_lr,2)~=size(ref_gray,2)
                    proc_lr = imresize(proc_lr, [size(ref_gray,1), size(ref_gray,2)]);
                end

                psnr_lr = psnr(proc_lr, ref_gray);
                ssim_lr = ssim(proc_lr, ref_gray);

            catch ME
                warning('Error al calcular PSNR/SSIM para "%s" vs. ref "%s": %s', ...
                    lr_name, ref_file, ME.message);
            end
        end

        % BRISQUE y PIQE (sin referencia)
        try
            lr_gray = lr_img_data;
            if size(lr_gray,3)==3
                lr_gray = rgb2gray(lr_gray);
            end
            brisque_lr = brisque(lr_gray);
            piqe_lr    = piqe(lr_gray);
        catch
            warning('No se pudo calcular BRISQUE/PIQE para "%s".', lr_name);
        end

        title(sprintf('LR/Deformed (%s)\nPSNR=%.2f, SSIM=%.2f, BRISQUE=%.2f, PIQE=%.2f', ...
            deform_str, psnr_lr, ssim_lr, brisque_lr, piqe_lr), 'Interpreter','none');

        % --------------------------------------------------
        % Tiles siguientes: cada modelo
        % --------------------------------------------------
        for m = 1:length(model_dirs)
            model_dir  = model_dirs{m};
            model_name = model_names{m};

            all_files  = dir(fullfile(model_dir, '*.*'));
            all_files  = all_files(~[all_files.isdir]);

            matched_file = '';
            % Buscamos un archivo que empiece con "<idx_str>_<deform_str>"
            pat = sprintf('^%s_%s', idx_str, deform_str);
            for ff = 1:length(all_files)
                if ~isempty(regexp(all_files(ff).name, pat, 'once'))
                    matched_file = all_files(ff).name;
                    break;
                end
            end

            nexttile(tiled_layout);
            if isempty(matched_file)
                imshow(uint8(ones(200,200,3)*128));
                title(sprintf('%s\n(No hallado)', model_name));
                continue;
            end

            proc_path = fullfile(model_dir, matched_file);
            try
                model_img = imread(proc_path);
                imshow(model_img);
            catch
                imshow(uint8(ones(200,200,3)*128));
                title(sprintf('%s\n(Error al leer)', model_name));
                continue;
            end

            % Métricas vs. ref
            [psnr_val, ssim_val, brisque_val, piqe_val] = deal(NaN);

            if has_reference
                try
                    ref_gray = ref_img;
                    if size(ref_gray,3)==3
                        ref_gray = rgb2gray(ref_gray);
                    end

                    proc_model = model_img;
                    if size(proc_model,3)==3
                        proc_model = rgb2gray(proc_model);
                    end

                    % Ajustamos tamaño
                    if size(proc_model,1)~=size(ref_gray,1) || ...
                       size(proc_model,2)~=size(ref_gray,2)
                        proc_model = imresize(proc_model, ...
                            [size(ref_gray,1), size(ref_gray,2)]);
                    end

                    psnr_val = psnr(proc_model, ref_gray);
                    ssim_val = ssim(proc_model, ref_gray);

                catch ME
                    warning('Error calculando PSNR/SSIM para "%s" vs. ref "%s": %s', ...
                            matched_file, ref_file, ME.message);
                end
            end

            % BRISQUE y PIQE
            try
                proc_gray = model_img;
                if size(proc_gray,3)==3
                    proc_gray = rgb2gray(proc_gray);
                end
                brisque_val = brisque(proc_gray);
                piqe_val    = piqe(proc_gray);
            catch
                warning('No se pudo calcular BRISQUE/PIQE para "%s".', matched_file);
            end

            title(sprintf('%s\nPSNR=%.2f, SSIM=%.2f\nBRISQUE=%.2f, PIQE=%.2f', ...
                model_name, psnr_val, ssim_val, brisque_val, piqe_val), 'Interpreter','none');
        end
        % Quedan 2 celdas libres (3x4=12), no pasa nada
    end
end
