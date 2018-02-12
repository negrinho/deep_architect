import tb_filesystem as fs
import tb_io as io
import subprocess

### project manipulation
def create_project_folder(folderpath, project_name):
    fn = lambda xs: fs.join_paths([folderpath, project_name] + xs)

    fs.create_folder( fn( [] ) )
    # typical directories
    fs.create_folder( fn( [project_name] ) )
    fs.create_folder( fn( ["analyses"] ) )     
    fs.create_folder( fn( ["data"] ) )
    fs.create_folder( fn( ["experiments"] ) )
    fs.create_folder( fn( ["notes"] ) )
    fs.create_folder( fn( ["temp"] ) )

    # code files (in order): data, preprocessing, model definition, model training, 
    # model evaluation, main to generate the results with different relevant 
    # parameters, setting up different experiments, analyze the results and 
    # generate plots and tables.
    fs.create_file( fn( [project_name, "__init__.py"] ) )
    fs.create_file( fn( [project_name, "data.py"] ) )
    fs.create_file( fn( [project_name, "preprocess.py"] ) )
    fs.create_file( fn( [project_name, "model.py"] ) )    
    fs.create_file( fn( [project_name, "train.py"] ) )
    fs.create_file( fn( [project_name, "evaluate.py"] ) ) 
    fs.create_file( fn( [project_name, "main.py"] ) )
    fs.create_file( fn( [project_name, "experiment.py"] ) )
    fs.create_file( fn( [project_name, "analyze.py"] ) )

    # add an empty script that can be used to download data.
    fs.create_file( fn( ["data", "download_data.py"] ) )

    # common notes to keep around.
    fs.create_file( fn( ["notes", "journal.txt"] ) )    
    fs.create_file( fn( ["notes", "reading_list.txt"] ) )    
    fs.create_file( fn( ["notes", "todos.txt"] ) )    

    # placeholders
    io.write_textfile( fn( ["experiments", "readme.txt"] ), 
        ["All experiments will be placed under this folder."] )

    io.write_textfile( fn( ["temp", "readme.txt"] ), 
        ["Here lie temporary files that are relevant or useful for the project "
        "but that are not kept under version control."] )

    io.write_textfile( fn( ["analyses", "readme.txt"] ), 
        ["Here lie files containing information extracted from the "
        "results of the experiments. Tables and plots are typical examples."] )

    # typical git ignore file.
    io.write_textfile( fn( [".gitignore"] ), 
        ["data", "experiments", "temp", "*.pyc", "*.pdf", "*.aux"] )
    
    subprocess.call("cd %s && git init && git add -f .gitignore * && "
        "git commit -a -m \"Initial commit for %s.\" && cd -" % ( 
            fn( [] ), project_name), shell=True)
    
