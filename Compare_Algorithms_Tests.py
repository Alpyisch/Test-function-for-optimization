import numpy as np
import time
import pandas as pd
import os
import subprocess
from Optimization_Functions import OptimizationFunctions

def run_algorithm(algorithm, function_name, dimension, population_size, max_iterations, trials=1):
    """
    Belirtilen algoritmayı belirtilen fonksiyon üzerinde çalıştırır ve sonuçları döndürür.
    
    Args:
        algorithm: Çalıştırılacak algoritma (ABC, PSO, DE, GEN)
        function_name: Optimize edilecek fonksiyon adı
        dimension: Problem boyutu
        population_size: Popülasyon boyutu
        max_iterations: Maksimum iterasyon sayısı
        trials: Deneme sayısı
        
    Returns:
        dict: Algoritmanın sonuçları
    """
    # Algoritma parametrelerini ayarla
    if algorithm == "ABC":
        cmd = [
            "python", "ABC.py",
            "--function", function_name,
            "--colony-size", str(population_size),
            "--dimension", str(dimension),
            "--max-cycles", str(max_iterations),
            "--trials", str(trials)
        ]
    elif algorithm == "PSO":
        cmd = [
            "python", "PSO.py",
            "--function", function_name,
            "--particles", str(population_size),
            "--dimension", str(dimension),
            "--max-iterations", str(max_iterations),
            "--trials", str(trials)
        ]
    elif algorithm == "DE":
        cmd = [
            "python", "DE.py",
            "--function", function_name,
            "--population", str(population_size),
            "--dimension", str(dimension),
            "--max-generations", str(max_iterations),
            "--trials", str(trials)
        ]
    elif algorithm == "GEN":
        cmd = [
            "python", "GEN.py",
            "--function", function_name,
            "--population-size", str(population_size),
            "--dimension", str(dimension),
            "--max-generations", str(max_iterations),
            "--trials", str(trials)
        ]
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algorithm}")
    
    # Algoritmayı çalıştır
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    execution_time = time.time() - start_time
    
    # Sonuçları analiz et
    if process.returncode != 0:
        print(f"Hata: {algorithm} algoritması {function_name} fonksiyonu üzerinde çalıştırılamadı.")
        print(f"Hata mesajı: {stderr}")
        return {
            "algorithm": algorithm,
            "function": function_name,
            "success": False,
            "error": stderr
        }
    
    # Çıktıyı analiz et
    result = {
        "algorithm": algorithm,
        "function": function_name,
        "success": True,
        "execution_time": execution_time
    }
    
    # Çıktıdan sonuçları çıkar
    if trials > 1:
        # Çoklu deneme sonuçları
        for line in stdout.split('\n'):
            if "Best Fitness Overall:" in line:
                result["best_fitness"] = float(line.split(":")[1].strip())
            elif "Mean Fitness:" in line:
                result["mean_fitness"] = float(line.split(":")[1].strip())
            elif "Std. Dev. Fitness:" in line:
                result["std_fitness"] = float(line.split(":")[1].strip())
            elif "Mean Cycles:" in line or "Mean Iterations:" in line or "Mean Generations:" in line:
                result["mean_iterations"] = float(line.split(":")[1].strip())
    else:
        # Tek deneme sonuçları
        for line in stdout.split('\n'):
            if "Best Fitness:" in line:
                result["best_fitness"] = float(line.split(":")[1].strip())
            elif "Cycles:" in line or "Iterations:" in line or "Generations:" in line:
                result["iterations"] = int(line.split(":")[1].strip())
    
    return result

def compare_algorithms(functions, dimension=2, population_size=50, max_iterations=1000, trials=1):
    """
    Tüm algoritmaları belirtilen fonksiyonlar üzerinde çalıştırır ve sonuçları karşılaştırır.
    
    Args:
        functions: Test edilecek fonksiyon adları listesi
        dimension: Problem boyutu
        population_size: Popülasyon boyutu
        max_iterations: Maksimum iterasyon sayısı
        trials: Deneme sayısı
        
    Returns:
        pandas.DataFrame: Karşılaştırma sonuçları
    """
    algorithms = ["ABC", "PSO", "DE", "GEN"]
    results = []
    
    for func in functions:
        print(f"\nFonksiyon: {func}")
        for algo in algorithms:
            print(f"  Algoritma: {algo}")
            result = run_algorithm(
                algorithm=algo,
                function_name=func,
                dimension=dimension,
                population_size=population_size,
                max_iterations=max_iterations,
                trials=trials
            )
            results.append(result)
    
    # Sonuçları DataFrame'e dönüştür
    df = pd.DataFrame(results)
    
    # Sonuçları Excel'e kaydet
    output_file = f"algorithm_comparison_results.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")
    
    return df

def get_available_functions():
    """
    Optimization_Functions.py'deki tüm fonksiyonları döndürür.
    """
    opt_functions = OptimizationFunctions()
    functions = []
    
    # Sınıfın tüm metodlarını al
    for method_name in dir(opt_functions):
        # Fonksiyon metodlarını filtrele
        if method_name.endswith('_function') and callable(getattr(opt_functions, method_name)):
            # Fonksiyon adını al (sondaki _function kısmını çıkar)
            func_name = method_name.replace('_function', '')
            functions.append(func_name)
    
    return functions

if __name__ == "__main__":
    # Test edilecek fonksiyonları al
    all_functions = get_available_functions()
    
    # Boyut gereksinimleri olan fonksiyonları filtrele
    dimension_requirements = {
        'sphere': 2,
        'ackley': 2,
        'three_hump_camel': 2,
        'six_hump_camel': 2,
        'dixon_price': 2,
        'rosenbrock': 2,
        'beale': 2,
        'branin': 2,
        'colville': 4,
        'forrester': 1,
        'goldstein_price': 2,
        'hartmann_3d': 3,
        'hartmann_4d': 4,
        'hartmann_6d': 6,
        'perm': 2,
        'powell': 4,
        'shekel': 4,
        'styblinski_tang': 2,
        'bukin': 2,
        'cross_in_tray': 2,
        'drop_wave': 2,
        'eggholder': 2,
        'gramacy_lee': 1,
        'griewank': 2,
        'holder_table': 2,
        'langermann': 2,
        'levy': 2,
        'levy_n13': 2,
        'rastrigin': 2,
        'schaffer_n2': 2,
        'schaffer_n4': 2,
        'schwefel': 2,
        'shubert': 2,
        'michalewicz': 2,
        'easom': 2,
        'booth': 2,
        'matyas': 2,
        'zakharov': 2,
        'bukin_n5': 2,
        'schwefel_226': 2,
        'sinc': 2,
        'ackley2': 2,
        'sum_squares': 2,
        'step': 2,
        'alpine': 2,
        'bukin_n4': 2,
        'cosine_mixture': 2
    }
    
    # Test edilecek fonksiyonları seç
    # Basitlik için sadece 2 boyutlu fonksiyonları test edelim
    test_functions = [func for func in all_functions if func in dimension_requirements and dimension_requirements[func] == 2]
    
    # Eğer çok fazla fonksiyon varsa, sadece ilk 10'unu al
    if len(test_functions) > 10:
        test_functions = test_functions[:10]
    
    print(f"Test edilecek fonksiyonlar: {test_functions}")
    
    # Algoritmaları karşılaştır
    results = compare_algorithms(
        functions=test_functions,
        dimension=2,
        population_size=50,
        max_iterations=1000,
        trials=1  # Hızlı sonuç için tek deneme
    )
    
    # Sonuçları göster
    print("\nKarşılaştırma Sonuçları:")
    print(results) 