using Documenter, MatrixEnsembles

format = Documenter.HTML(edit_link = "master",
                         prettyurls = get(ENV, "CI", nothing) == "true",
)

makedocs(
    clean = true,
    format = format,
    sitename = "CuQuantumInformation.jl",
    authors = "Łukasz Pawela",
    assets = ["assets/favicon.ico"],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "man/randomqobjects.md",
        ],
        "Library" => "lib/CuQuantumInformation.md"
    ]
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    target = "build",
    repo = "github.com/iitis/CuQuantumInformation.jl.git"
)
