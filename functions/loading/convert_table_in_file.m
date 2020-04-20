function data_out = convert_table_in_file(file)

data = load(file);
data_out = correct_table_in_data(data);

if isstruct(data_out)
    save([file(1:end-4),'_FR.mat'],'-struct','data_out')
else
    save([file(1:end-4),'_FR.mat'],'data_out')
end
end

function data_new = correct_table_in_data(data)

if isstruct(data)
    data_fields = fields(data);
    for field = 1:numel(data_fields)
        data.(data_fields{field}) = correct_table_in_data(data.(data_fields{field}));
    end
    data_new = data;
elseif istable(data)
    data_new = struct();
    data_fields = fields(data);
    for field = 1:numel(data_fields)
        data_new.(data_fields{field}) = data.(data_fields{field});
    end
elseif iscategorical(data)
    data_new = string(data);
else
    data_new = data;
end

end