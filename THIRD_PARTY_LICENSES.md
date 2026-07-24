# Third-Party Licenses

AegisRouter 使用了以下开源项目的源码（vendored 方式引入，锁定版本）。

---

## LiteLLM

- **路径**: `vendor/litellm/`
- **许可证**: MIT License
- **项目地址**: https://github.com/BerriAI/litellm
- **用途**: LLM 代理网关骨架，提供 OpenAI 兼容 API、多模型管理、Failover 等能力

### MIT License

```
MIT License

Copyright (c) 2023 BerriAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## RouteLLM

- **路径**: `vendor/routellm/`
- **许可证**: Apache License 2.0
- **项目地址**: https://github.com/lm-sys/RouteLLM
- **用途**: Prompt 难度评估分类器，本地推理输出复杂度分数用于智能路由

### Apache License 2.0

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

## ClawVault

- **路径**: `vendor/clawvault/`
- **许可证**: MIT License
- **项目地址**: https://github.com/nickelghost/claw-vault
- **用途**: PII 数据保险库，提供脱敏/还原基础框架，AegisRouter 在此基础上扩展中文 PII 识别和 UDS 通信

### MIT License

```
MIT License

Copyright (c) 2024 ClawVault Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 合规说明

所有 vendor 源码均保留原始 LICENSE 文件。本项目对 vendor 源码的修改均以二次开发方式在
`aegis_router/` 目录下实现，不直接修改 vendor 源文件（除非必要的 bugfix）。
