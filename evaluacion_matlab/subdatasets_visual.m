function comparaModelos3_Visual

    % =====================================================================
    % 1) Definir rutas de los 3 modelos y de las carpetas originales
    % =====================================================================

    % Carpeta base de los modelos DRTC, HAT, RealESRGAN
    base_path = 'C:\Users\nekos\OneDrive\Escritorio\resultados_subdataset\resultados_subdataset';

    model_names = {'DRTC','HAT','RealESRGAN'};
    model_dirs = {
        fullfile(base_path, 'DRTC'), ...
        fullfile(base_path, 'HAT'),  ...
        fullfile(base_path, 'RealESRGAN')
    };

    % Carpeta base de originales, con "COLOR", "ELEMENTOS", etc.
    originals_base = 'C:\Users\nekos\OneDrive\Escritorio\originals';

    % Dos categorías
    top_level_folders = {'COLOR','ELEMENTOS'};


    % =====================================================================
    % 2) Crear la figura y el uitabgroup
    % =====================================================================
    fig = figure('Name','Comparación SR (3 modelos)','NumberTitle','off',...
                 'WindowState','maximized');
    tab_group = uitabgroup(fig);


    % =====================================================================
    % 3) Recorrer cada categoría (COLOR, ELEMENTOS)
    % =====================================================================
    for c = 1:numel(top_level_folders)
        cat_name = top_level_folders{c};
        cat_folder = fullfile(originals_base, cat_name);

        if ~exist(cat_folder,'dir')
            warning('No existe la carpeta de originales: %s', cat_folder);
            continue;
        end

        % -----------------------------------------------------------------
        % 3.1) Buscar todas las subcarpetas en DRTC, HAT, RealESRGAN
        %      (dentro de "COLOR" o "ELEMENTOS")
        % -----------------------------------------------------------------
        all_subfolders = {};
        for m = 1:numel(model_dirs)
            this_cat_path = fullfile(model_dirs{m}, cat_name);
            if ~exist(this_cat_path,'dir'), continue; end

            listing = dir(this_cat_path);
            for k = 1:numel(listing)
                if listing(k).isdir && ~strcmp(listing(k).name,'.') && ~strcmp(listing(k).name,'..')
                    all_subfolders{end+1} = listing(k).name; %#ok<AGROW>
                end
            end
        end

        % Extraer el "suffix" eliminando "DRTC_", "HAT_", "RealESRGAN_"
        suff_list = cell(size(all_subfolders));
        for iS = 1:numel(all_subfolders)
            sf = all_subfolders{iS};
            if     startsWith(sf,'DRTC_'),        sf = erase(sf,'DRTC_');
            elseif startsWith(sf,'HAT_'),         sf = erase(sf,'HAT_');
            elseif startsWith(sf,'RealESRGAN_'),  sf = erase(sf,'RealESRGAN_');
            end
            suff_list{iS} = sf;
        end
        unique_suffs = unique(suff_list);

        % -----------------------------------------------------------------
        % 3.2) Para cada sufijo, mostrar su primera imagen + Ref + LR
        % -----------------------------------------------------------------
        for sfx = 1:numel(unique_suffs)
            suffix_name = unique_suffs{sfx};

            % 1) Hallar un índice de imagen (p.ej. "22") buscando
            %    la PRIMERA imagen en la subcarpeta DRTC, HAT o RealESRGAN
            [index_found, idx_num] = findIndexInAnyModel(model_dirs, model_names, cat_name, suffix_name);
            if ~index_found
                continue; % No hay imágenes => no creamos pestaña
            end

            % Crear una pestaña
            tab_title = sprintf('%s - %s (idx=%d)', cat_name, suffix_name, idx_num);
            the_tab = uitab(tab_group,'Title',tab_title);

            tgrid = tiledlayout(the_tab,2,3,'TileSpacing','loose','Padding','loose');

            % 2) Cargar la Referencia
            [ref_img, ref_title] = loadReference(cat_name, suffix_name, idx_num, originals_base);

            % 3) Cargar la LR/Deformada
            [lr_img, lr_title] = loadLR(cat_name, suffix_name, idx_num, originals_base);

            % -------------------------------------------------------------
            % TILE (1): Referencia
            % -------------------------------------------------------------
            nexttile(tgrid,1);
            if isempty(ref_img)
                imshow(uint8(ones(200,200,3)*128));
                title('Referencia no encontrada');
            else
                imshow(ref_img);
                title(ref_title, 'Interpreter','none');
            end

            % -------------------------------------------------------------
            % TILE (2): LR
            % -------------------------------------------------------------
            nexttile(tgrid,2);
            if isempty(lr_img)
                imshow(uint8(ones(200,200,3)*128));
                title('LR no encontrada');
            else
                imshow(lr_img);
                [ps,ss,br,pq] = computeMetrics(lr_img, ref_img);
                title(sprintf('LR\nPSNR=%.2f, SSIM=%.2f\nBRISQ=%.1f, PIQE=%.1f', ...
                    ps, ss, br, pq), 'Interpreter','none');
            end

            % -------------------------------------------------------------
            % TILEs (3,4,5): DRTC, HAT, RealESRGAN
            % -------------------------------------------------------------
            for mm = 1:numel(model_names)
                nexttile(tgrid, 2+mm);  % 3,4,5

                % Subcarpeta: "DRTC_BlancoNegroBlur", etc.
                this_model_subfolder = sprintf('%s_%s', model_names{mm}, suffix_name);
                this_model_path = fullfile(model_dirs{mm}, cat_name, this_model_subfolder);

                if ~exist(this_model_path,'dir')
                    imshow(uint8(ones(200,200,3)*128));
                    title(sprintf('%s (no dir)', model_names{mm}));
                    continue;
                end

                % Tomamos la PRIMERA imagen
                flist = dir(fullfile(this_model_path,'*.*'));
                flist = flist(~[flist.isdir]);
                if isempty(flist)
                    imshow(uint8(ones(200,200,3)*128));
                    title(sprintf('%s (vacío)', model_names{mm}));
                    continue;
                end

                sr_path = fullfile(this_model_path, flist(1).name);
                try
                    sr_img = imread(sr_path);
                    imshow(sr_img);

                    [ps,ss,br,pq] = computeMetrics(sr_img, ref_img);
                    title(sprintf('%s\nPSNR=%.2f, SSIM=%.2f\nBRISQ=%.1f, PIQE=%.1f', ...
                        model_names{mm}, ps, ss, br, pq), 'Interpreter','none');
                catch
                    imshow(uint8(ones(200,200,3)*128));
                    title(sprintf('%s\nError al leer', model_names{mm}));
                end
            end
        end
    end
end


% =========================================================================
% findIndexInAnyModel: busca la 1ª imagen en (DRTC|HAT|RealESRGAN),
% extrae el índice (p.ej. "22") del inicio del nombre y lo devuelve.
% =========================================================================
function [found_flag, idx_num] = findIndexInAnyModel(model_dirs, model_names, cat_name, suffix_name)
    found_flag = false;
    idx_num    = 0;

    for mm = 1:numel(model_names)
        subf = sprintf('%s_%s', model_names{mm}, suffix_name);
        path_m = fullfile(model_dirs{mm}, cat_name, subf);
        if ~exist(path_m,'dir'), continue; end

        flist = dir(fullfile(path_m,'*.*'));
        flist = flist(~[flist.isdir]);
        if isempty(flist), continue; end

        % Tomamos la primera imagen y extraemos su índice
        test_name = flist(1).name; % p.ej. "22.jpg"
        tokens = regexp(test_name, '^(\d+)', 'tokens');
        if ~isempty(tokens)
            idx_num = str2double(tokens{1}{1});
            found_flag = true;
            return;
        end
    end
end

% =========================================================================
% loadReference: construye la carpeta de referencia (p.ej. "BlancoNegro\BlancoNegroOriginal")
% y busca idx_num.jpg, etc.
% =========================================================================
function [ref_img, ref_title] = loadReference(cat_name, suffix_name, idx, originals_base)
    ref_img   = [];
    ref_title = 'Referencia';

    ref_folder = buildRefFolder(suffix_name, cat_name, originals_base);  % p.ej. ...\COLOR\BlancoNegro\BlancoNegroOriginal
    if ~exist(ref_folder,'dir'), return; end

    exts = {'.jpg','.png','.bmp','.tif'};
    for e = 1:numel(exts)
        candidate = fullfile(ref_folder, sprintf('%d%s', idx, exts{e}));
        if exist(candidate,'file')
            try
                ref_img = imread(candidate);
                ref_title = sprintf('Ref: %d%s', idx, exts{e});
                return;
            catch
            end
        end
    end
end

% =========================================================================
% loadLR: construye la carpeta LR (p.ej. "BlancoNegro\BlancoNegroOriginalBlur")
% =========================================================================
function [lr_img, lr_title] = loadLR(cat_name, suffix_name, idx, originals_base)
    lr_img   = [];
    lr_title = 'LR/Deformed';

    lr_folder = buildLRFolder(suffix_name, cat_name, originals_base);
    if ~exist(lr_folder,'dir'), return; end

    exts = {'.jpg','.png','.bmp','.tif'};
    for e = 1:numel(exts)
        candidate = fullfile(lr_folder, sprintf('%d%s', idx, exts{e}));
        if exist(candidate,'file')
            try
                lr_img = imread(candidate);
                lr_title = sprintf('LR: %d%s', idx, exts{e});
                return;
            catch
            end
        end
    end
end

% =========================================================================
% buildRefFolder: decide "BlancoNegroOriginal", "RetratoOriginal", etc.
% =========================================================================
function ref_folder = buildRefFolder(suffix_name, cat_name, originals_base)
    if contains(lower(suffix_name), 'blanconegro')
        subName = 'BlancoNegro'; 
        baseRef = 'BlancoNegroOriginal';
    elseif contains(lower(suffix_name), 'cartoon')
        subName = 'Cartoon';     
        baseRef = 'CartoonOriginal';
    elseif contains(lower(suffix_name), 'dibujo')
        subName = 'Dibujo';      
        baseRef = 'DibujoOriginal';
    elseif contains(lower(suffix_name), 'fondoscargados')
        subName = 'FondosCargados';
        baseRef = 'FondoscargadosOriginal';
    elseif contains(lower(suffix_name), 'minimalista')
        subName = 'Minimalista';
        baseRef = 'MinimalistaOriginal';
    elseif contains(lower(suffix_name), 'sepia')
        subName = 'Sepia';
        baseRef = 'SepiaOriginal';
    elseif contains(lower(suffix_name), 'retrato')
        subName = 'Retrato';
        baseRef = 'RetratoOriginal';
    elseif contains(lower(suffix_name), 'color')
        subName = 'Color';
        baseRef = 'ColorOriginal';
    else
        subName = 'Desconocido';
        baseRef = 'DesconocidoOriginal';
    end

    ref_folder = fullfile(originals_base, cat_name, subName, baseRef);
end

% =========================================================================
% buildLRFolder: decide "BlancoNegroOriginalBlur", etc.
% =========================================================================
function lr_folder = buildLRFolder(suffix_name, cat_name, originals_base)
    % Determina la parte base (ej. "BlancoNegroOriginal") + deformación
    if contains(lower(suffix_name), 'blanconegro')
        subName = 'BlancoNegro';
        baseName = 'BlancoNegroOriginal';
    elseif contains(lower(suffix_name), 'cartoon')
        subName = 'Cartoon';
        baseName = 'CartoonOriginal';
    elseif contains(lower(suffix_name), 'dibujo')
        subName = 'Dibujo';
        baseName = 'DibujoOriginal';
    elseif contains(lower(suffix_name), 'fondoscargados')
        subName = 'FondosCargados';
        baseName = 'FondoscargadosOriginal';
    elseif contains(lower(suffix_name), 'minimalista')
        subName = 'Minimalista';
        baseName = 'MinimalistaOriginal';
    elseif contains(lower(suffix_name), 'sepia')
        subName = 'Sepia';
        baseName = 'SepiaOriginal';
    elseif contains(lower(suffix_name), 'retrato')
        subName = 'Retrato';
        baseName = 'RetratoOriginal';
    elseif contains(lower(suffix_name), 'color')
        subName = 'Color';
        baseName = 'ColorOriginal';
    else
        subName = 'Desconocido';
        baseName = 'DesconocidoOriginal';
    end

    % Extraer la parte "Blur", "Noise", "Resized", etc.
    pat = lower(subName);
    remain = erase(lower(suffix_name), pat);  % p.ej. "blur"
    remain = strtrim(remain);

    % Evitar duplicar "original"
    if startsWith(remain,'original')
        remain = erase(remain,'original');
    end

    lr_sub = [baseName, remain];  % p.ej. "BlancoNegroOriginalBlur"
    lr_folder = fullfile(originals_base, cat_name, subName, lr_sub);
end

% =========================================================================
% computeMetrics: PSNR/SSIM (con ref), BRISQUE/PIQE (sin ref).
% =========================================================================
function [psnr_val, ssim_val, brisque_val, piqe_val] = computeMetrics(proc_img, ref_img)
    psnr_val = NaN; ssim_val = NaN;
    brisque_val = NaN; piqe_val = NaN;

    % Sin referencia
    if ~isempty(proc_img)
        try
            proc_gray = rgb2gray(proc_img);
            brisque_val = brisque(proc_gray);
            piqe_val    = piqe(proc_gray);
        catch
        end
    end

    % Con referencia
    if ~isempty(proc_img) && ~isempty(ref_img)
        try
            ref_gray = ref_img;
            if size(ref_gray,3)==3
                ref_gray = rgb2gray(ref_gray);
            end
            proc_gray = rgb2gray(proc_img);

            % Ajustar tamaño
            if size(proc_gray,1)~=size(ref_gray,1) || size(proc_gray,2)~=size(ref_gray,2)
                proc_gray = imresize(proc_gray,[size(ref_gray,1), size(ref_gray,2)]);
            end

            psnr_val = psnr(proc_gray, ref_gray);
            ssim_val = ssim(proc_gray, ref_gray);
        catch
        end
    end
end
