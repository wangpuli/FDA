function file_name = get_system_file_name(xds)
monkey = strcat(xds.meta.monkey, '_');
date = xds.meta.dateTime;
y = datestr( date, 'yyyy' );
m = datestr( date, 'mm' );
d = datestr( date, 'dd');
date_str = strcat(y, strcat(m, d));
serial = xds.meta.rawFileName(end-3:end);
file_name = strcat(monkey, strcat(date_str, serial));
disp(file_name);
end

