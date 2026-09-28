using System.Runtime.InteropServices;
using GlueVision.Core;
using GlueVision.Hardware;
using GlueVision.Hardware.Mock;
using GlueVision.Process.Coordinate;
using GlueVision.Process.Running;
using GlueVision.Vision;

// 让 CLR 找到 halcon.dll 原生库
string? halconRoot = Environment.GetEnvironmentVariable("HALCONROOT");
if (!string.IsNullOrEmpty(halconRoot))
{
    string? arch = Environment.GetEnvironmentVariable("HALCONARCH");
    if (string.IsNullOrEmpty(arch)) arch = "x64-win64";
    string dir = Path.Combine(halconRoot, "bin", arch);
    if (Directory.Exists(dir)) SetDllDirectory(dir);
}

int failures = 0;
void Check(string name, bool ok, string detail = "")
{
    Console.WriteLine($"[{(ok ? "PASS" : "FAIL")}] {name}  {detail}");
    if (!ok) failures++;
}

// ---------- 1. 硬件/视觉装配 ----------
var cfg = new SystemConfig
{
    MotionBackend = MotionBackend.Mock,
    CameraBackend = CameraBackend.Mock,
    VisionBackend = VisionBackend.Halcon,   // 有 Halcon 用 Halcon, 无则工厂自动回退
    SafeZ = 20,
    WorkZ = 0
};

using var bundle = HardwareFactory.CreateBundle(cfg);
var motion = bundle.Motion;
var camera = bundle.Camera;
var valve = bundle.Valve;
var stage = bundle.Stage!;

IVisionService vision;
try
{
    vision = VisionFactory.Create(VisionBackend.Halcon, out string? notice);
    if (!string.IsNullOrEmpty(notice)) Console.WriteLine("NOTICE: " + notice);
}
catch (Exception ex)
{
    Console.WriteLine("Halcon 后端不可用, 回退 Mock: " + ex.Message);
    vision = VisionFactory.Create(VisionBackend.Mock, out _);
}
Console.WriteLine($"视觉后端: {vision.Backend} - {vision.BackendInfo}");

string modelDir = Path.Combine(Path.GetTempPath(), "GlueVisionSmoke", "Models");
Directory.CreateDirectory(modelDir);
string mark1File = Path.Combine(modelDir, "mark1.shm");
string mark2File = Path.Combine(modelDir, "mark2.shm");

// ---------- 2. 连接 + 回零 ----------
await motion.ConnectAsync("sim");
await camera.ConnectAsync(new CameraParams());
Check("设备连接", motion.IsConnected && camera.IsConnected);

await motion.HomeAsync();
Check("三轴回零", motion.IsHomed);

// ---------- 3. 配方(双 Mark + 矩形胶路) ----------
var recipe = new Recipe { Name = "SmokeProduct" };
recipe.Marks.Add(new MarkDefinition
{
    Name = "Mark1", ModelFile = mark1File,
    ReferenceMachine = new Point2(100, 100),
    TrainParam = new ModelTrainParam { Row = 512, Col = 640, HalfWidth = 70, HalfHeight = 70, MinScore = 0.6 }
});
recipe.Marks.Add(new MarkDefinition
{
    Name = "Mark2", ModelFile = mark2File,
    ReferenceMachine = new Point2(160, 100),
    TrainParam = new ModelTrainParam { Row = 512, Col = 640, HalfWidth = 70, HalfHeight = 70, MinScore = 0.6 }
});
// 胶路旋转中心必须与仿真产品旋转中心(ProductOrigin=100,100)一致, 才能对比补偿轨迹
recipe.Path.Origin = new Point2(100, 100);
recipe.Path.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(100, 80), End = new(160, 80) });
recipe.Path.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(160, 80), End = new(160, 120) });
recipe.Path.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(160, 120), End = new(100, 120) });
recipe.Path.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(100, 120), End = new(100, 80) });

// ---------- 4. 训练两个 Mark 模板(生产场景) ----------
stage.Scene = SimScene.Production;
VisionImage GrabAt(double x, double y)
{
    motion.MoveZAsync(cfg.SafeZ, cfg.RapidSpeed).GetAwaiter().GetResult();
    motion.MoveXYAsync(x, y, cfg.RapidSpeed).GetAwaiter().GetResult();
    motion.MoveZAsync(cfg.WorkZ, cfg.RapidSpeed).GetAwaiter().GetResult();
    Thread.Sleep(120);
    return camera.GrabOne();
}

var roiParam = new ModelTrainParam { Row = 512, Col = 640, HalfWidth = 70, HalfHeight = 70, MinScore = 0.6 };
vision.TrainShapeModel(GrabAt(100, 100), roiParam, mark1File);
vision.TrainShapeModel(GrabAt(160, 100), roiParam, mark2File);
Check("双 Mark 模板训练", File.Exists(mark1File) && File.Exists(mark2File));

var testFind = vision.FindShapeModel(GrabAt(100, 100), mark1File, 0.6, -15, 30);
Check("模板匹配(零偏差)", testFind.Found,
    testFind.Found ? $"row={testFind.Row:F2} col={testFind.Col:F2} ang={testFind.AngleDeg:F3} score={testFind.Score:F2}" : testFind.Message);

// ---------- 5. 九点标定(标定场景) ----------
stage.Scene = SimScene.Calibration;
var calibSession = new NinePointCalibSession(vision);
var grid = NinePointCalibSession.GenerateGrid(stage.FixedCalibMark, 10, 10);
int foundCount = 0;
foreach (var p in grid)
{
    var img = GrabAt(p.X, p.Y);
    var r = vision.FindShapeModel(img, mark1File, 0.6, -15, 30);
    if (r.Found) { calibSession.AddPoint(p.X, p.Y, r.Row, r.Col); foundCount++; }
}
Check("九点采集(9/9)", foundCount == 9, $"实际 {foundCount} 点");
var calib = calibSession.Build();
Check("九点标定 RMS<0.02mm", calib.IsValid && calib.RmsErrorMm < 0.02, $"RMS={calib.RmsErrorMm:F5}mm");

// 标定正确性独立验证：固定 Mark 在机器 (80,80), 相机移到网格内未采集点 (85,85), 像素换算应还原
{
    var img = GrabAt(85, 85);
    var r = vision.FindShapeModel(img, mark1File, 0.6, -15, 30);
    if (r.Found)
    {
        var center = calib.ToMachine(512, 640);
        var feat = calib.ToMachine(r.Row, r.Col);
        // 移动相机标定法: 特征 = 相机位 - (g(q) - g(center))
        double fx = 85 - (feat.X - center.X);
        double fy = 85 - (feat.Y - center.Y);
        double err = Math.Sqrt((fx - 80) * (fx - 80) + (fy - 80) * (fy - 80));
        Check("标定点坐标还原误差<0.03mm", err < 0.03, $"还原=({fx:F3},{fy:F3}) err={err:F4}mm");
    }
    else Check("标定点坐标还原误差<0.03mm", false, "未找到 Mark");
}
stage.Scene = SimScene.Production;
await motion.MoveZAsync(cfg.SafeZ, cfg.RapidSpeed);

// ---------- 6. 设置产品偏差并自动跑一片 ----------
stage.ShiftX = 1.5;
stage.ShiftY = -1.0;
stage.ShiftThetaDeg = 0.8;
Console.WriteLine($"注入产品偏差: dX=1.5mm dY=-1.0mm dθ=0.8°");

var runner = new GlueJobRunner(motion, camera, valve, vision, cfg)
{
    CurrentRecipe = recipe,
    Calibration = calib
};

LocateResult? located = null;
var headPoints = new List<Point2>();
var firstDone = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
JobState? terminalState = null;
runner.Located += (_, e) => located = e.Result;
runner.StepProgress += (_, e) => headPoints.Add(e.HeadPosition);
runner.StateChanged += (_, e) =>
{
    if (e.State is JobState.Done or JobState.Alarm)
    {
        terminalState ??= e.State;
        firstDone.TrySetResult();
    }
};
runner.Message += (_, m) => Console.WriteLine("  [job] " + m);

runner.Start();
var finished = await Task.WhenAny(firstDone.Task, Task.Delay(TimeSpan.FromSeconds(120)));
runner.Stop();
await Task.Delay(300);

Check("自动片完成(Done)", finished == firstDone.Task && terminalState == JobState.Done,
    $"完成状态={terminalState?.ToString() ?? "超时"}");
Check("视觉定位成功", located is { Success: true }, located?.Message ?? "无定位结果");
if (located is { Success: true })
{
    Check("dX 偏差精度<0.05mm", Math.Abs(located.OffsetX - 1.5) < 0.05, $"测得 dX={located.OffsetX:F4}");
    Check("dY 偏差精度<0.05mm", Math.Abs(located.OffsetY - (-1.0)) < 0.05, $"测得 dY={located.OffsetY:F4}");
    Check("dθ 偏差精度<0.1°", Math.Abs(located.OffsetThetaDeg - 0.8) < 0.1, $"测得 dθ={located.OffsetThetaDeg:F4}°");
}
Check("产量 OK=1 / NG=0", runner.Stats.OK == 1 && runner.Stats.NG == 0, $"OK={runner.Stats.OK} NG={runner.Stats.NG}");

// 补偿轨迹验证：第一段起点(100,80) 按注入真值绕(100,100)转0.8°再平移(1.5,-1)，应出现在实际轨迹中
{
    var expect = MathUtil.Rotate(new Point2(100, 80), 0.8, new Point2(100, 100));
    expect = new Point2(expect.X + 1.5, expect.Y - 1.0);
    double minDist = headPoints.Count == 0 ? double.MaxValue :
        headPoints.Min(p => Math.Sqrt((p.X - expect.X) * (p.X - expect.X) + (p.Y - expect.Y) * (p.Y - expect.Y)));
    Check("补偿后轨迹命中<0.6mm", minDist < 0.6,
        $"期望=({expect.X:F3},{expect.Y:F3}) 最近距离={minDist:F3}mm 轨迹点={headPoints.Count}");
}

Check("胶阀最终关闭", !valve.IsOpen);
Console.WriteLine($"\n周期: {runner.Stats.LastCycleMs} ms");

// ---------- 7. 胶路采样器静态验证 ----------
{
    var p = new GluePath();
    p.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(100, 80), End = new(160, 80) });
    p.Segments.Add(new GlueSegment { Type = SegmentType.Dot, Start = new(160, 80), DwellMs = 100 });
    p.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(160, 80), End = new(160, 120) });
    var s = GluePathSampler.Build(p, new GlueProcessSettings());
    // Dot 结尾已关阀+抬Z, 第三段必须重新趋近(修复前"趋近"只有 1 次)
    Check("Dot打断连续胶路(第三段重新趋近)", s.Count(x => x.Remark == "趋近") == 2,
        $"趋近步数={s.Count(x => x.Remark == "趋近")}");

    var p2 = new GluePath();
    p2.Segments.Add(new GlueSegment { Type = SegmentType.Line, Start = new(0, 0), End = new(1, 0) });
    var s2 = GluePathSampler.Build(p2, new GlueProcessSettings());
    Check("短线段(1mm<LeadClose0.8)仍出胶", s2.Any(x => x.GlueOn && x.Remark == "点胶"));
}

// ---------- 8. 多轮偏差浸泡 + 暂停/继续 ----------
(double dx, double dy, double dt)[] rounds = [(2.0, 1.5, -1.2), (-2.5, 1.0, 1.5), (0.5, -2.0, -0.6)];
for (int r = 0; r < rounds.Length; r++)
{
    stage.ShiftX = rounds[r].dx;
    stage.ShiftY = rounds[r].dy;
    stage.ShiftThetaDeg = rounds[r].dt;
    var rr = new GlueJobRunner(motion, camera, valve, vision, cfg)
    {
        CurrentRecipe = recipe,
        Calibration = calib
    };
    LocateResult? loc = null;
    var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
    rr.Located += (_, e) => loc = e.Result;
    rr.StateChanged += (_, e) =>
    {
        if (e.State is JobState.Done or JobState.Alarm) done.TrySetResult();
    };
    rr.Start();

    if (r == 1) // 第二轮中途验证暂停/继续
    {
        await Task.Delay(4000);
        rr.Pause();
        await Task.Delay(500);
        bool pausedOk = rr.State == JobState.Paused;
        rr.Resume();
        Check("暂停/继续", pausedOk, $"暂停时状态={rr.State}");
    }

    await Task.WhenAny(done.Task, Task.Delay(TimeSpan.FromSeconds(60)));
    rr.Stop();
    await Task.Delay(200);

    Check($"第{r + 1}轮定位成功(d={rounds[r].dx},{rounds[r].dy},{rounds[r].dt}°)",
        loc is { Success: true }, loc?.Message ?? "超时/无结果");
    if (loc is { Success: true })
    {
        Check($"第{r + 1}轮偏差精度",
            Math.Abs(loc.OffsetX - rounds[r].dx) < 0.05
            && Math.Abs(loc.OffsetY - rounds[r].dy) < 0.05
            && Math.Abs(loc.OffsetThetaDeg - rounds[r].dt) < 0.1,
            $"测得 dX={loc.OffsetX:F3} dY={loc.OffsetY:F3} dθ={loc.OffsetThetaDeg:F3}°");
    }
    Check($"第{r + 1}轮胶阀关闭", !valve.IsOpen);
}

Console.WriteLine(failures == 0 ? "\n===== SMOKE TEST PASSED =====" : $"\n===== {failures} 项失败 =====");
return failures == 0 ? 0 : 1;

[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
static extern bool SetDllDirectory(string lpPathName);
