using LMDiskANN
using Documenter

DocMeta.setdocmeta!(LMDiskANN, :DocTestSetup, :(using LMDiskANN); recursive=true)

makedocs(;
    modules=[LMDiskANN],
    authors="Alexander V. Mantzaris",
    sitename="LMDiskANN.jl",
    format=Documenter.HTML(;
        canonical="https://mantzaris.github.io/LMDiskANN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mantzaris/LMDiskANN.jl",
    devbranch="main",
)
